using Microsoft.ML.OnnxRuntime;
using SciencePcm.Embed;

namespace SciencePcm.Server;

public sealed class SharedReranker : IDisposable
{
    private readonly InferenceSession _session;
    private readonly ThreadLocal<ICrossEncoder> _encoders;
    private readonly SemaphoreSlim _slots;
    private readonly int _batchSize;

    public SharedReranker(ServerOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxConcurrentReranks, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.RerankBatch, 1);
        _batchSize = options.RerankBatch;
        _session = TextEmbedder.CreateSession(options.CrossEncoderPath, options.Threads,
            options.UseGpu, deviceId: 0, gpuMemLimitBytes: options.GpuMemoryLimitBytes);
        _slots = new SemaphoreSlim(options.MaxConcurrentReranks, options.MaxConcurrentReranks);
        _encoders = new ThreadLocal<ICrossEncoder>(() => CrossEncoderFactory.Create(
            new CrossEncoderOptions(options.CrossEncoderPath, options.Threads, options.MaxTokens),
            _session), trackAllValues: true);
    }

    public float[] Score(string query, IReadOnlyList<string> passages)
    {
        _slots.Wait();
        try
        {
            var encoder = _encoders.Value!;
            var scores = new float[passages.Count];
            for (var start = 0; start < passages.Count; start += _batchSize)
            {
                var batch = passages.Skip(start).Take(_batchSize).ToArray();
                var batchScores = encoder.Score(query, batch);
                Array.Copy(batchScores, 0, scores, start, batchScores.Length);
            }
            return scores;
        }
        finally
        {
            _slots.Release();
        }
    }

    public void Dispose()
    {
        foreach (var encoder in _encoders.Values) encoder.Dispose();
        _encoders.Dispose();
        _slots.Dispose();
        _session.Dispose();
    }
}