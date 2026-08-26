using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SciencePcm.Embed;

public sealed record CrossEncoderOptions(
    string ModelDirectory,
    int IntraOpThreads = 8,
    int MaxTokens = 512,
    bool LowerCase = true,
    TokenizerKind Tokenizer = TokenizerKind.Fast,
    int PadMultiple = 32);

/// <summary>
/// Scores a query against passages. Implementations differ in how they tokenise, which
/// is the part that varies between model families.
/// </summary>
public interface ICrossEncoder : IDisposable
{
    float[] Score(string query, IReadOnlyList<string> passages);
}

/// <summary>
/// MedCPT cross-encoder: scores a query against passages by reading both together.
/// Slower than a bi-encoder because nothing can be precomputed, but the LLM judge
/// measured it as the single largest quality win (nDCG@10 0.686 -> 0.790).
/// </summary>
public sealed class CrossEncoder : ICrossEncoder
{
    private static readonly string[] InputNames = ["input_ids", "attention_mask", "token_type_ids"];
    private const string OutputName = "logits";

    private readonly InferenceSession _session;
    private readonly bool _ownsSession;
    private readonly ICorpusTokenizer _tokenizer;
    private readonly CrossEncoderOptions _options;
    private readonly int _separatorId;
    private readonly bool _needsTokenTypeIds;

    public CrossEncoder(CrossEncoderOptions options, InferenceSession? sharedSession = null)
    {
        _options = options;
        _session = sharedSession ?? TextEmbedder.CreateSession(options.ModelDirectory, options.IntraOpThreads);
        _ownsSession = sharedSession is null;
        _needsTokenTypeIds = _session.InputMetadata.ContainsKey("token_type_ids");
        _tokenizer = TokenizerFactory.Create(
            options.ModelDirectory, options.Tokenizer, options.MaxTokens, options.LowerCase);

        // Encoding the empty string yields [CLS] [SEP], which is how the separator id is
        // discovered without hardcoding a vocabulary offset.
        var empty = _tokenizer.Encode("");
        _separatorId = empty.Length >= 2
            ? empty[^1]
            : throw new InvalidOperationException("Tokenizer did not emit [CLS]/[SEP] for empty input.");
    }

    /// <summary>
    /// Builds [CLS] query [SEP] passage [SEP]. FastBertTokenizer has no pair overload, so
    /// the two halves are encoded separately and joined; the passage's leading [CLS] is
    /// dropped. Segment ids are 0 across the query half and 1 across the passage half.
    /// </summary>
    public (int[] Ids, int[] TypeIds) EncodePair(string query, string passage)
    {
        var queryIds = _tokenizer.Encode(query ?? "");
        var passageIds = _tokenizer.Encode(passage ?? "");

        if (queryIds.Length >= _options.MaxTokens)
        {
            queryIds = [.. queryIds.Take(_options.MaxTokens - 1), _separatorId];
        }

        var body = passageIds.AsSpan(1);
        var available = _options.MaxTokens - queryIds.Length;

        // Truncating the passage rather than the query matches HuggingFace's
        // longest_first behaviour here, where the passage is always the longer half.
        var truncated = body.Length > available;
        if (truncated)
        {
            body = body[..Math.Max(0, available - 1)];
        }

        var ids = new int[queryIds.Length + body.Length + (truncated ? 1 : 0)];
        queryIds.CopyTo(ids, 0);
        body.CopyTo(ids.AsSpan(queryIds.Length));
        if (truncated) ids[^1] = _separatorId;

        var typeIds = new int[ids.Length];
        for (var i = queryIds.Length; i < typeIds.Length; i++) typeIds[i] = 1;

        return (ids, typeIds);
    }

    public float[] Score(string query, IReadOnlyList<string> passages)
    {
        if (passages.Count == 0) return [];

        var encoded = new (int[] Ids, int[] TypeIds)[passages.Count];
        var longest = 0;
        for (var i = 0; i < passages.Count; i++)
        {
            encoded[i] = EncodePair(query, passages[i]);
            longest = Math.Max(longest, encoded[i].Ids.Length);
        }

        // Rounding the padded width to a multiple keeps the number of distinct input
        // shapes small, which stops the GPU allocator fragmenting over a long run.
        var width = Math.Min(
            _options.MaxTokens,
            (int)(Math.Ceiling(longest / (double)_options.PadMultiple) * _options.PadMultiple));
        width = Math.Max(width, longest);

        var batch = passages.Count;
        var inputIds = new DenseTensor<long>([batch, width]);
        var attention = new DenseTensor<long>([batch, width]);
        var types = _needsTokenTypeIds ? new DenseTensor<long>([batch, width]) : null;

        for (var row = 0; row < batch; row++)
        {
            var (ids, typeIds) = encoded[row];
            for (var column = 0; column < ids.Length; column++)
            {
                inputIds[row, column] = ids[column];
                attention[row, column] = 1;
                if (types is not null) types[row, column] = typeIds[column];
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputNames[0], inputIds),
            NamedOnnxValue.CreateFromTensor(InputNames[1], attention),
        };
        if (types is not null)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(InputNames[2], types));
        }

        using var results = _session.Run(inputs);
        var logits = results.First(r => r.Name == OutputName).AsTensor<float>();

        var scores = new float[batch];
        for (var row = 0; row < batch; row++) scores[row] = logits[row, 0];
        return scores;
    }

    /// <summary>
    /// Replays the pair samples in tokenizer-parity.json. The pair is assembled by hand,
    /// so a divergence here would silently degrade every reranked result.
    /// </summary>
    public static (int Passed, int Total, List<string> Failures) VerifyParity(string parityPath, CrossEncoder encoder)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(parityPath));
        var samples = document.RootElement.GetProperty("samples");

        var failures = new List<string>();
        var passed = 0;
        var total = 0;

        foreach (var sample in samples.EnumerateArray())
        {
            total++;
            var query = sample.GetProperty("query").GetString() ?? "";
            var passage = sample.GetProperty("passage").GetString() ?? "";
            var expectedIds = sample.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var expectedTypes = sample.GetProperty("token_type_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();

            var (ids, typeIds) = encoder.EncodePair(query, passage);

            if (ids.SequenceEqual(expectedIds) && typeIds.SequenceEqual(expectedTypes))
            {
                passed++;
                continue;
            }

            failures.Add(
                $"\"{query}\" + \"{passage[..Math.Min(40, passage.Length)]}...\"\n" +
                $"    expected ids   : {string.Join(",", expectedIds)}\n" +
                $"    actual ids     : {string.Join(",", ids)}\n" +
                $"    expected types : {string.Join(",", expectedTypes)}\n" +
                $"    actual types   : {string.Join(",", typeIds)}");
        }

        return (passed, total, failures);
    }

    public void Dispose()
    {
        if (_ownsSession) _session.Dispose();
    }
}
