using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace SciencePcm.Embed;

public enum PoolingMode
{
    /// <summary>First-token embedding. What MedCPT and most BERT retrievers use.</summary>
    Cls,
    MeanOverAttended,
}

public sealed record EmbedderOptions(
    string ModelDirectory,
    int IntraOpThreads = 8,
    int MaxTokens = 512,
    PoolingMode Pooling = PoolingMode.Cls,
    bool Normalize = true,
    bool LowerCase = true,
    TokenizerKind Tokenizer = TokenizerKind.Fast);

/// <summary>
/// One ONNX session plus tokenizer. Not thread-safe: create one per worker so that
/// N modest sessions run concurrently rather than one session trying to use every
/// core, which collapses under scheduling contention on a high-core-count machine.
/// </summary>
public sealed class TextEmbedder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly bool _ownsSession;
    private readonly ICorpusTokenizer _tokenizer;
    private readonly string[] _inputNames;
    private readonly bool _needsTokenTypeIds;
    private readonly string _outputName;
    private readonly bool _outputIsAlreadyPooled;
    private readonly EmbedderOptions _options;

    public int Dimensions { get; }

    /// <summary>Padded token count of the most recent batch, for throughput accounting.</summary>
    public long LastBatchTokens { get; private set; }

    public TextEmbedder(EmbedderOptions options, InferenceSession? sharedSession = null)
    {
        _options = options;

        var modelPath = ResolveModelPath(options.ModelDirectory);

        if (sharedSession is not null)
        {
            _session = sharedSession;
            _ownsSession = false;
        }
        else
        {
            _session = CreateSession(modelPath, options.IntraOpThreads);
            _ownsSession = true;
        }

        _tokenizer = TokenizerFactory.Create(
            options.ModelDirectory, options.Tokenizer, options.MaxTokens, options.LowerCase);

        _inputNames = [.. _session.InputMetadata.Keys];
        _needsTokenTypeIds = _session.InputMetadata.ContainsKey("token_type_ids");

        _outputName = _session.OutputMetadata.ContainsKey("last_hidden_state")
            ? "last_hidden_state"
            : _session.OutputMetadata.Keys.First();
        _outputIsAlreadyPooled = _session.OutputMetadata[_outputName].Dimensions.Length == 2;

        Dimensions = Embed(["probe"])[0].Length;
    }

    /// <summary>
    /// Run() is thread-safe, so one session can back many workers. Creating a session per
    /// worker duplicates the weights, which wrecks cache locality and NUMA placement.
    /// </summary>
    public static InferenceSession CreateSession(string modelDirectory, int intraOpThreads, bool useGpu = false, int deviceId = 0)
    {
        var modelPath = File.Exists(modelDirectory) ? modelDirectory : ResolveModelPath(modelDirectory);

        var sessionOptions = new SessionOptions
        {
            IntraOpNumThreads = intraOpThreads,
            InterOpNumThreads = 1,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };

        if (useGpu)
        {
#if USE_GPU
            sessionOptions.AppendExecutionProvider_CUDA(deviceId);
#else
            throw new NotSupportedException(
                "CUDA support is not compiled in. Rebuild with: dotnet build -c Release -p:UseGpu=true");
#endif
        }
        else
        {
            // Spin-waiting threads fight each other when many sessions share a socket.
            sessionOptions.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
        }

        return new InferenceSession(modelPath, sessionOptions);
    }

    private static string ResolveModelPath(string directory)
    {
        foreach (var candidate in new[] { "model.onnx", "model_quantized.onnx", "model_int8.onnx" })
        {
            var path = Path.Combine(directory, candidate);
            if (File.Exists(path)) return path;
        }

        var any = Directory.EnumerateFiles(directory, "*.onnx").FirstOrDefault();
        return any ?? throw new FileNotFoundException($"No .onnx file found in {directory}.");
    }

    /// <summary>Token ids for one text, used by the parity check.</summary>
    public IReadOnlyList<int> Tokenize(string text) => _tokenizer.Encode(text);

    public float[][] Embed(IReadOnlyList<string> texts)
    {
        var batch = texts.Count;
        var encoded = new IReadOnlyList<int>[batch];
        var longest = 1;

        for (var i = 0; i < batch; i++)
        {
            encoded[i] = _tokenizer.Encode(texts[i] ?? "");
            longest = Math.Max(longest, encoded[i].Count);
        }

        // Pad only to the longest member of this batch, not to MaxTokens. Abstracts are
        // far shorter than 512 tokens, so this is most of the throughput.
        var ids = new DenseTensor<long>([batch, longest]);
        var mask = new DenseTensor<long>([batch, longest]);
        var types = _needsTokenTypeIds ? new DenseTensor<long>([batch, longest]) : null;
        LastBatchTokens = (long)batch * longest;

        for (var row = 0; row < batch; row++)
        {
            var tokens = encoded[row];
            for (var col = 0; col < tokens.Count; col++)
            {
                ids[row, col] = tokens[col];
                mask[row, col] = 1;
            }
        }

        var inputs = new List<NamedOnnxValue>(3)
        {
            NamedOnnxValue.CreateFromTensor("input_ids", ids),
            NamedOnnxValue.CreateFromTensor("attention_mask", mask),
        };
        if (types is not null)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", types));
        }

        using var results = _session.Run(inputs, [_outputName]);
        var tensor = results.First().AsTensor<float>();

        return _outputIsAlreadyPooled
            ? ExtractPooled(tensor, batch)
            : Pool(tensor, mask, batch, longest);
    }

    private float[][] ExtractPooled(Tensor<float> tensor, int batch)
    {
        var hidden = tensor.Dimensions[1];
        var output = new float[batch][];
        for (var row = 0; row < batch; row++)
        {
            var vector = new float[hidden];
            for (var d = 0; d < hidden; d++) vector[d] = tensor[row, d];
            if (_options.Normalize) NormalizeInPlace(vector);
            output[row] = vector;
        }
        return output;
    }

    private float[][] Pool(Tensor<float> tensor, DenseTensor<long> mask, int batch, int length)
    {
        var hidden = tensor.Dimensions[2];
        var output = new float[batch][];

        for (var row = 0; row < batch; row++)
        {
            var vector = new float[hidden];

            if (_options.Pooling == PoolingMode.Cls)
            {
                for (var d = 0; d < hidden; d++) vector[d] = tensor[row, 0, d];
            }
            else
            {
                var counted = 0;
                for (var token = 0; token < length; token++)
                {
                    if (mask[row, token] == 0) continue;
                    counted++;
                    for (var d = 0; d < hidden; d++) vector[d] += tensor[row, token, d];
                }
                if (counted > 0)
                {
                    for (var d = 0; d < hidden; d++) vector[d] /= counted;
                }
            }

            if (_options.Normalize) NormalizeInPlace(vector);
            output[row] = vector;
        }

        return output;
    }

    private static void NormalizeInPlace(float[] vector)
    {
        double sum = 0;
        foreach (var value in vector) sum += value * value;
        if (sum <= 0) return;

        var scale = (float)(1.0 / Math.Sqrt(sum));
        for (var i = 0; i < vector.Length; i++) vector[i] *= scale;
    }

    /// <summary>Records the exact configuration used, so query-time can assert it matches.</summary>
    public JsonObjectSnapshot Describe() => new(
        Path.GetFileName(ResolveModelPath(_options.ModelDirectory)),
        Dimensions,
        _options.MaxTokens,
        _options.Pooling.ToString(),
        _options.Normalize,
        _options.LowerCase,
        _needsTokenTypeIds,
        _outputName);

    public void Dispose()
    {
        if (_ownsSession) _session.Dispose();
    }
}

public sealed record JsonObjectSnapshot(
    string ModelFile,
    int Dimensions,
    int MaxTokens,
    string Pooling,
    bool Normalized,
    bool LowerCase,
    bool UsesTokenTypeIds,
    string OutputName)
{
    public string ToJson() => JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
}
