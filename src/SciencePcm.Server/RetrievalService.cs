using Microsoft.ML.OnnxRuntime;
using SciencePcm.Embed;
using SciencePcm.Lexical;

namespace SciencePcm.Server;

public sealed record ServerOptions
{
    public required string IndexPath { get; init; }
    public required string CrossEncoderPath { get; init; }
    public int RerankCandidates { get; init; } = 100;
    public int RerankBatch { get; init; } = 32;
    public int MaxTokens { get; init; } = 512;
    public int Threads { get; init; } = 8;
    public bool UseGpu { get; init; }
    public long GpuMemoryLimitBytes { get; init; }
}

public sealed record SearchResult(
    string ArticleKey,
    string Title,
    string Abstract,
    int Year,
    string Pmid,
    float Score,
    string Stage);

/// <summary>
/// BM25 retrieval followed by cross-encoder reranking.
///
/// This is what the LLM judge picked: reranking lifted graded nDCG@10 from 0.676 to
/// 0.771, while dense retrieval and RRF fusion added nothing measurable on top of it.
/// BioASQ ranked these the other way round, but only ~10% of its judgements exist in
/// this corpus, so it scored good unjudged papers as failures.
/// </summary>
public sealed class RetrievalService : IDisposable
{
    private readonly LexicalSearcher _lexical;
    private readonly InferenceSession _session;
    private readonly ThreadLocal<CrossEncoder> _rerankers;
    private readonly ServerOptions _options;

    public RetrievalService(ServerOptions options)
    {
        _options = options;
        _lexical = new LexicalSearcher(options.IndexPath);

        _session = TextEmbedder.CreateSession(
            options.CrossEncoderPath,
            options.Threads,
            options.UseGpu,
            deviceId: 0,
            gpuMemLimitBytes: options.GpuMemoryLimitBytes);

        // Run() is thread-safe so the session is shared, but each tokenizer is not.
        _rerankers = new ThreadLocal<CrossEncoder>(() => new CrossEncoder(
            new CrossEncoderOptions(options.CrossEncoderPath, options.Threads, options.MaxTokens),
            _session));
    }

    public int DocumentCount => _lexical.Count;

    public IReadOnlyList<SearchResult> Search(
        string query,
        int k,
        int? yearMin = null,
        int? yearMax = null,
        bool rerank = true)
    {
        var candidates = _lexical.SearchArticles(
            query,
            rerank ? Math.Max(k, _options.RerankCandidates) : k,
            yearMin,
            yearMax);

        if (candidates.Count == 0) return [];

        if (!rerank)
        {
            return Deduplicate(candidates.Select(c => ToResult(c, c.Score, "bm25"))).Take(k).ToList();
        }

        var encoder = _rerankers.Value!;
        var scores = new float[candidates.Count];

        for (var start = 0; start < candidates.Count; start += _options.RerankBatch)
        {
            var slice = candidates.Skip(start).Take(_options.RerankBatch).ToList();
            var passages = slice
                .Select(c => string.IsNullOrEmpty(c.Title) ? c.Body : c.Title + ". " + c.Body)
                .ToList();

            var batchScores = encoder.Score(query, passages);
            Array.Copy(batchScores, 0, scores, start, batchScores.Length);
        }

        var reranked = candidates
            .Select((hit, index) => ToResult(hit, scores[index], "bm25+rerank"))
            .OrderByDescending(r => r.Score);

        return Deduplicate(reranked).Take(k).ToList();
    }

    public SearchResult? GetPaper(string articleKey)
    {
        var hit = _lexical.GetByKey(articleKey);
        return hit is null ? null : ToResult(hit, 0f, "lookup");
    }

    private static SearchResult ToResult(LexicalHit hit, float score, string stage) =>
        new(hit.ArticleKey, hit.Title, hit.Body, hit.Year, hit.Pmid, score, stage);

    /// <summary>
    /// The corpus contains the same paper under several OpenAlex ids - the qualitative
    /// review turned up "Friedreich ataxia." twice in one top-10. Callers should never
    /// see that, so equal titles collapse to the best-scoring copy.
    /// </summary>
    private static IEnumerable<SearchResult> Deduplicate(IEnumerable<SearchResult> results)
    {
        var seenTitles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var seenKeys = new HashSet<string>(StringComparer.Ordinal);

        foreach (var result in results)
        {
            if (!seenKeys.Add(result.ArticleKey)) continue;

            var title = result.Title.Trim().TrimEnd('.');
            if (title.Length > 0 && !seenTitles.Add(title)) continue;

            yield return result;
        }
    }

    public void Dispose()
    {
        foreach (var encoder in _rerankers.Values) encoder.Dispose();
        _rerankers.Dispose();
        _session.Dispose();
        _lexical.Dispose();
    }
}
