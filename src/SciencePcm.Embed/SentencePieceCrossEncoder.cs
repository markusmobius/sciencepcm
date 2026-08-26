using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SciencePcm.Embed;

/// <summary>Special token ids, written beside the model by tools/export_onnx.py.</summary>
public sealed record SpecialTokens(int Bos, int Eos, int Pad, int Unk)
{
    public static SpecialTokens Load(string modelDirectory)
    {
        var path = Path.Combine(modelDirectory, "special-tokens.json");
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"No special-tokens.json in {modelDirectory}. Re-export with tools/export_onnx.py.", path);
        }

        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        return new SpecialTokens(
            root.GetProperty("bos_token_id").GetInt32(),
            root.GetProperty("eos_token_id").GetInt32(),
            root.GetProperty("pad_token_id").GetInt32(),
            root.GetProperty("unk_token_id").GetInt32());
    }
}

/// <summary>
/// Cross-encoder for XLM-RoBERTa models such as BAAI/bge-reranker-v2-m3.
///
/// Those use SentencePiece, which FastBertTokenizer cannot do, so tokenisation runs as
/// its own ONNX graph built by onnxruntime-extensions. That keeps a single tokenizer
/// implementation - the one HuggingFace exported - rather than reimplementing
/// SentencePiece in C# and hoping it matches.
///
/// The tokenizer graph handles one sequence at a time, so pairs are assembled here as
/// XLM-R expects: bos A eos eos B eos, and no token_type_ids.
/// </summary>
public sealed class SentencePieceCrossEncoder : ICrossEncoder
{
    private readonly InferenceSession _tokenizerSession;
    private readonly InferenceSession _session;
    private readonly bool _ownsSession;
    private readonly CrossEncoderOptions _options;
    private readonly SpecialTokens _special;

    public SentencePieceCrossEncoder(CrossEncoderOptions options, InferenceSession? sharedSession = null)
    {
        _options = options;
        _special = SpecialTokens.Load(options.ModelDirectory);

        var tokenizerPath = Path.Combine(options.ModelDirectory, "tokenizer.onnx");
        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException($"No tokenizer.onnx in {options.ModelDirectory}.", tokenizerPath);
        }

        // Custom ops live in the extensions library, and only this session needs them.
        var tokenizerOptions = new SessionOptions();
        tokenizerOptions.RegisterOrtExtensions();
        _tokenizerSession = new InferenceSession(tokenizerPath, tokenizerOptions);

        _session = sharedSession ?? TextEmbedder.CreateSession(options.ModelDirectory, options.IntraOpThreads);
        _ownsSession = sharedSession is null;
    }

    private int[] Tokenize(string text)
    {
        var input = new DenseTensor<string>([1]);
        input[0] = text ?? "";

        using var results = _tokenizerSession.Run(
            [NamedOnnxValue.CreateFromTensor(_tokenizerSession.InputMetadata.Keys.First(), input)]);

        var ids = results.First().AsTensor<long>();
        var trimmed = new List<int>((int)ids.Length);

        // The graph emits the special tokens for a single sequence; pair assembly needs
        // the bare pieces, so drop them here and add the pair layout below.
        foreach (var id in ids.ToArray())
        {
            var value = (int)id;
            if (value == _special.Bos || value == _special.Eos || value == _special.Pad) continue;
            trimmed.Add(value);
        }

        return [.. trimmed];
    }

    public int[] EncodePair(string query, string passage)
    {
        var q = Tokenize(query);
        var p = Tokenize(passage);

        // bos A eos eos B eos
        var overhead = 4;
        var budget = _options.MaxTokens - overhead;
        if (q.Length > budget / 2) q = [.. q.Take(Math.Max(1, budget / 2))];

        var remaining = Math.Max(0, budget - q.Length);
        if (p.Length > remaining) p = [.. p.Take(remaining)];

        var ids = new int[q.Length + p.Length + overhead];
        var cursor = 0;
        ids[cursor++] = _special.Bos;
        foreach (var id in q) ids[cursor++] = id;
        ids[cursor++] = _special.Eos;
        ids[cursor++] = _special.Eos;
        foreach (var id in p) ids[cursor++] = id;
        ids[cursor++] = _special.Eos;

        return ids;
    }

    public float[] Score(string query, IReadOnlyList<string> passages)
    {
        if (passages.Count == 0) return [];

        var encoded = new int[passages.Count][];
        var longest = 0;
        for (var i = 0; i < passages.Count; i++)
        {
            encoded[i] = EncodePair(query, passages[i]);
            longest = Math.Max(longest, encoded[i].Length);
        }

        var width = Math.Max(
            longest,
            Math.Min(_options.MaxTokens,
                (int)(Math.Ceiling(longest / (double)_options.PadMultiple) * _options.PadMultiple)));

        var batch = passages.Count;
        var inputIds = new DenseTensor<long>([batch, width]);
        var attention = new DenseTensor<long>([batch, width]);

        for (var row = 0; row < batch; row++)
        {
            for (var column = 0; column < width; column++)
            {
                var inRange = column < encoded[row].Length;
                inputIds[row, column] = inRange ? encoded[row][column] : _special.Pad;
                attention[row, column] = inRange ? 1 : 0;
            }
        }

        using var results = _session.Run(
        [
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attention),
        ]);

        var logits = results.First(r => r.Name == "logits").AsTensor<float>();
        var scores = new float[batch];
        for (var row = 0; row < batch; row++) scores[row] = logits[row, 0];
        return scores;
    }

    public void Dispose()
    {
        _tokenizerSession.Dispose();
        if (_ownsSession) _session.Dispose();
    }
}

public static class CrossEncoderFactory
{
    /// <summary>
    /// A tokenizer.onnx beside the model means SentencePiece; otherwise WordPiece via
    /// FastBertTokenizer. So --cross-encoder alone selects the implementation.
    /// </summary>
    public static ICrossEncoder Create(CrossEncoderOptions options, InferenceSession? sharedSession = null) =>
        File.Exists(Path.Combine(options.ModelDirectory, "tokenizer.onnx"))
            ? new SentencePieceCrossEncoder(options, sharedSession)
            : new CrossEncoder(options, sharedSession);

    public static string Describe(string modelDirectory) =>
        File.Exists(Path.Combine(modelDirectory, "tokenizer.onnx"))
            ? "SentencePiece (tokenizer.onnx)"
            : "WordPiece (FastBertTokenizer)";
}
