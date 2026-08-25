using System.ComponentModel;
using System.Text.Json;
using ModelContextProtocol.Server;

namespace SciencePcm.Server;

[McpServerToolType]
public sealed class LiteratureTools(RetrievalService retrieval)
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    [McpServerTool(Name = "search_literature")]
    [Description(
        "Search a corpus of neuroscience paper abstracts and return the most relevant papers. " +
        "Use a full natural-language question rather than keywords; the reranker reads the " +
        "question and the abstract together, so phrasing carries information.")]
    public string SearchLiterature(
        [Description("The research question, in natural language.")] string query,
        [Description("How many papers to return. Default 10, maximum 50.")] int k = 10,
        [Description("Only include papers published in or after this year.")] int? yearMin = null,
        [Description("Only include papers published in or before this year.")] int? yearMax = null,
        [Description("Skip reranking. Faster, noticeably less relevant. Default false.")] bool fast = false)
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            return "{\"error\": \"query must not be empty\"}";
        }

        var results = retrieval.Search(query, Math.Clamp(k, 1, 50), yearMin, yearMax, rerank: !fast);

        return JsonSerializer.Serialize(new
        {
            query,
            returned = results.Count,
            results = results.Select(r => new
            {
                article_key = r.ArticleKey,
                title = r.Title,
                year = r.Year,
                pmid = string.IsNullOrEmpty(r.Pmid) ? null : r.Pmid,
                score = Math.Round(r.Score, 3),
                // Truncated deliberately: full abstracts for 10 papers would crowd out the
                // caller's context. get_paper returns the whole thing on request.
                abstract_excerpt = Excerpt(r.Abstract, 400),
            }),
        }, Json);
    }

    [McpServerTool(Name = "get_paper")]
    [Description("Retrieve one paper in full by its article key, as returned by search_literature.")]
    public string GetPaper(
        [Description("The article key, for example https://openalex.org/W2154021234")] string articleKey)
    {
        var paper = retrieval.GetPaper(articleKey);
        if (paper is null)
        {
            return JsonSerializer.Serialize(new { error = "not found", article_key = articleKey }, Json);
        }

        return JsonSerializer.Serialize(new
        {
            article_key = paper.ArticleKey,
            title = paper.Title,
            year = paper.Year,
            pmid = string.IsNullOrEmpty(paper.Pmid) ? null : paper.Pmid,
            @abstract = paper.Abstract,
        }, Json);
    }

    [McpServerTool(Name = "corpus_stats")]
    [Description(
        "Describe what this corpus covers and how it was built. Call this before drawing " +
        "conclusions from an absence of results.")]
    public string CorpusStats()
    {
        return JsonSerializer.Serialize(new
        {
            documents = retrieval.DocumentCount,
            tier = "abstracts",
            source = "OpenAlex, filtered to any topic in field 28 (neuroscience)",
            retrieval = "BM25 (Lucene EnglishAnalyzer) then MedCPT cross-encoder reranking",
            caveats = new[]
            {
                "The field-28 filter is broad, so some papers are only tangentially neuroscience.",
                "Abstracts only. Full text is not searchable in this tier.",
                "Absence of a paper here does not mean it does not exist.",
            },
        }, Json);
    }

    private static string Excerpt(string text, int limit)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= limit) return text;
        var cut = text.LastIndexOf(' ', Math.Min(limit, text.Length - 1));
        return text[..(cut > 0 ? cut : limit)] + " ...";
    }
}
