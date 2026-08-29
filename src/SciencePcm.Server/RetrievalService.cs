using Microsoft.ML.OnnxRuntime;
using SciencePcm.Embed;
using SciencePcm.Lexical;

namespace SciencePcm.Server;

public sealed record ServerOptions
{
    public required string IndexPath { get; init; }
    public required string CrossEncoderPath { get; init; }
    public string? PassageIndexPath { get; init; }
    public int RerankCandidates { get; init; } = 100;
    public int RerankBatch { get; init; } = 32;
    public int MaxTokens { get; init; } = 512;
    public int Threads { get; init; } = 8;
    public bool ParallelSearch { get; init; } = true;
    public double MaxDocFreqRatio { get; init; }

    /// <summary>
    /// Work types dropped before ranking. OpenAlex records peer reviews, datasets and
    /// front matter as works in their own right, titled after the paper they discuss, so
    /// they crowd out the paper itself when the paper has no abstract of its own.
    /// </summary>
    public IReadOnlyCollection<string> ExcludeWorkTypes { get; init; } = [];

    /// <summary>
    /// Weight on log10(1 + citations), added to the rerank score. A news article names
    /// papers people talk about, so citations break ties between a landmark paper and the
    /// commentaries that share its title. 0 disables the prior.
    /// </summary>
    public double CitationPriorWeight { get; init; }

    /// <summary>BM25 length normalisation. Lower values penalise long documents less.</summary>
    public float Bm25B { get; init; } = 0.75f;
    public bool UseGpu { get; init; }
    public long GpuMemoryLimitBytes { get; init; }
}

public sealed record SearchResult(
    string Id,
    string ArticleKey,
    string Title,
    string Text,
    int Year,
    string Pmid,
    string Section,
    bool IsRetracted,
    float Score,
    string Stage,
    BibliographicMetadata? Metadata = null);

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
    private readonly LexicalSearcher? _passages;
    private readonly InferenceSession _session;
    private readonly ThreadLocal<ICrossEncoder> _rerankers;
    private readonly ServerOptions _options;

    public RetrievalService(ServerOptions options)
    {
        _options = options;
        _lexical = new LexicalSearcher(
            options.IndexPath, 4, options.ParallelSearch, options.MaxDocFreqRatio,
            options.CitationPriorWeight, options.Bm25B);
        _passages = string.IsNullOrWhiteSpace(options.PassageIndexPath)
            ? null
            : new LexicalSearcher(
                options.PassageIndexPath, 4, options.ParallelSearch, options.MaxDocFreqRatio);

        _session = TextEmbedder.CreateSession(
            options.CrossEncoderPath,
            options.Threads,
            options.UseGpu,
            deviceId: 0,
            gpuMemLimitBytes: options.GpuMemoryLimitBytes);

        // Run() is thread-safe so the session is shared, but each tokenizer is not.
        _rerankers = new ThreadLocal<ICrossEncoder>(() => CrossEncoderFactory.Create(
            new CrossEncoderOptions(options.CrossEncoderPath, options.Threads, options.MaxTokens),
            _session));
    }

    public int DocumentCount => _lexical.Count;

    public int PassageCount => _passages?.Count ?? 0;

    /// <summary>Papers with full text, as opposed to passages.</summary>
    public long FullTextArticleCount => _passages?.DistinctArticleCount ?? 0;

    public bool HasFullText => _passages is not null;

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
            yearMax,
            _options.ExcludeWorkTypes);

        if (candidates.Count == 0) return [];

        if (!rerank)
        {
            return Deduplicate(candidates.Select(c => ToResult(c, c.Score, "bm25"))).Take(k).ToList();
        }

        var scores = Rerank(query, candidates);

        var reranked = candidates
            .Select((hit, index) => ToResult(hit, (float)(scores[index] + CitationPrior(hit)), "bm25+rerank"))
            .OrderByDescending(r => r.Score);

        return Deduplicate(reranked).Take(k).ToList();
    }

    /// <summary>
    /// Citation counts are missing or wrong on duplicate records, so this only ever adds -
    /// a paper with no recorded citations keeps its rerank score rather than being pushed
    /// down.
    /// </summary>
    private double CitationPrior(LexicalHit hit)
    {
        if (_options.CitationPriorWeight <= 0) return 0;
        var citations = hit.Metadata?.CitedByCount ?? 0;
        return citations <= 0 ? 0 : _options.CitationPriorWeight * Math.Log10(1 + citations);
    }

    public SearchResult? GetPaper(string articleKey)
    {
        var hit = _lexical.GetByKey(articleKey);
        return hit is null ? null : ToResult(hit, 0f, "lookup");
    }

    /// <summary>
    /// Passage-level search over full text. Returns fragments rather than whole articles:
    /// the abstracts tier already answers "which paper", and a cross-encoder can only
    /// score 512 tokens at a time, so a whole article is not a rankable unit.
    /// </summary>
    public IReadOnlyList<SearchResult> SearchFullText(
        string query,
        int k,
        string? section = null,
        int? yearMin = null,
        int? yearMax = null,
        bool includeRetracted = true,
        int maxPerArticle = 2)
    {
        if (_passages is null) return [];

        var candidates = _passages.Search(
            query,
            Math.Max(k, _options.RerankCandidates),
            yearMin,
            yearMax,
            section,
            includeRetracted);

        if (candidates.Count == 0) return [];

        var scores = Rerank(query, candidates);
        var ranked = candidates
            .Select((hit, index) => ToResult(hit, scores[index], "bm25+rerank"))
            .OrderByDescending(r => r.Score);

        return CapPerArticle(ranked, Math.Max(1, maxPerArticle)).Take(k).ToList();
    }

    /// <summary>Neighbouring passages, for when a fragment cuts mid-argument.</summary>
    public IReadOnlyList<SearchResult> GetPassageContext(string chunkId, int before, int after)
    {
        if (_passages is null) return [];

        // ChunkId is "{ArticleKey}#{index:D4}", so neighbours are addressable directly
        // rather than needing a range query.
        var hash = chunkId.LastIndexOf('#');
        if (hash < 0 || !int.TryParse(chunkId[(hash + 1)..], out var index)) return [];

        var articleKey = chunkId[..hash];
        var results = new List<SearchResult>();

        for (var offset = -Math.Abs(before); offset <= Math.Abs(after); offset++)
        {
            var neighbour = index + offset;
            if (neighbour < 0) continue;

            var hit = _passages.GetById($"{articleKey}#{neighbour:D4}");
            if (hit is not null) results.Add(ToResult(hit, 0f, "context"));
        }

        return results;
    }

    private float[] Rerank(string query, IReadOnlyList<LexicalHit> candidates)
    {
        var encoder = _rerankers.Value!;
        var scores = new float[candidates.Count];

        for (var start = 0; start < candidates.Count; start += _options.RerankBatch)
        {
            var slice = candidates.Skip(start).Take(_options.RerankBatch).ToList();
            var passages = slice.Select(RerankText).ToList();

            var batchScores = encoder.Score(query, passages);
            Array.Copy(batchScores, 0, scores, start, batchScores.Length);
        }

        return scores;
    }

    private static string RerankText(LexicalHit hit)
    {
        if (hit.Metadata is not { } metadata)
        {
            return string.IsNullOrEmpty(hit.Title) ? hit.Body : hit.Title + ". " + hit.Body;
        }

        // Keep newspaper-style identity clues visible inside the 512-token window,
        // but cap prolific author/institution lists so the abstract still has room.
        return string.Join(". ", new[]
        {
            hit.Title,
            Limit(metadata.Authors, 400),
            Limit(metadata.Journal, 200),
            Limit(metadata.Institutions, 400),
            Limit(metadata.Doi, 150),
            Limit(metadata.Topics, 300),
            hit.Body,
        }.Where(value => !string.IsNullOrWhiteSpace(value)));
    }

    private static string Limit(string value, int length) =>
        value.Length <= length ? value : value[..length];

    /// <summary>One thorough paper should not be able to fill the whole result set.</summary>
    private static IEnumerable<SearchResult> CapPerArticle(IEnumerable<SearchResult> results, int limit)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var result in results)
        {
            counts.TryGetValue(result.ArticleKey, out var seen);
            if (seen >= limit) continue;

            counts[result.ArticleKey] = seen + 1;
            yield return result;
        }
    }

    private static SearchResult ToResult(LexicalHit hit, float score, string stage) =>
        new(hit.Id, hit.ArticleKey, hit.Title, hit.Body, hit.Year, hit.Pmid, hit.Section, hit.IsRetracted, score, stage, hit.Metadata);

    /// <summary>
    /// The corpus contains the same paper under several OpenAlex ids - the qualitative
    /// review turned up "Friedreich ataxia." twice in one top-10. Callers should never
    /// see that, so equal titles collapse to the best-scoring copy.
    /// </summary>
    private static IEnumerable<SearchResult> Deduplicate(IEnumerable<SearchResult> results)
    {
        var ordered = results as IList<SearchResult> ?? results.ToList();

        // OpenAlex holds the same paper under the same title more than once - a preprint,
        // a stub typed "other", a merged-but-not-removed record - and the copies carry
        // few or no citations. Keeping whichever copy scored highest silently discarded
        // the canonical record, so pick the best of each title group by citations and
        // give it the group's best position.
        var best = new Dictionary<string, SearchResult>(StringComparer.OrdinalIgnoreCase);
        foreach (var result in ordered)
        {
            var key = TitleKey(result);
            if (key.Length == 0) continue;
            if (!best.TryGetValue(key, out var incumbent) || Citations(result) > Citations(incumbent))
            {
                best[key] = result;
            }
        }

        var seenTitles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var seenKeys = new HashSet<string>(StringComparer.Ordinal);

        foreach (var result in ordered)
        {
            var key = TitleKey(result);
            if (key.Length == 0)
            {
                if (seenKeys.Add(result.ArticleKey)) yield return result;
                continue;
            }

            if (!seenTitles.Add(key)) continue;
            var winner = best[key];
            if (seenKeys.Add(winner.ArticleKey)) yield return winner;
        }
    }

    private static string TitleKey(SearchResult result) => result.Title.Trim().TrimEnd('.');

    private static int Citations(SearchResult result) => result.Metadata?.CitedByCount ?? 0;

    public void Dispose()
    {
        foreach (var encoder in _rerankers.Values) encoder.Dispose();
        _rerankers.Dispose();
        _session.Dispose();
        _passages?.Dispose();
        _lexical.Dispose();
    }
}
