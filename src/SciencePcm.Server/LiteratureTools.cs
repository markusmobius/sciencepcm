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
        "Search ABSTRACTS of neuroscience papers and return the most relevant papers. This is " +
        "the tier to use for breadth - which papers exist on a topic, what a field says. " +
        "For specific findings, methods or numbers inside a paper, use search_full_text. " +
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
                abstract_excerpt = Excerpt(r.Text, 400),
            }),
        }, Json);
    }

    [McpServerTool(Name = "search_full_text")]
    [Description(
        "Search the FULL TEXT of neuroscience papers and return matching passages, not whole " +
        "papers. Use this for what a study actually did or found: methods, sample sizes, " +
        "parameters, specific results. Each hit is a ~300 word fragment labelled with its " +
        "section, so it can be quoted and attributed. " +
        "IMPORTANT: full text covers only about 23% of the corpus, so absence here does not " +
        "mean absence from the literature - fall back to search_literature for breadth. " +
        "Retracted papers ARE included and flagged is_retracted; say so if you cite one.")]
    public string SearchFullText(
        [Description("The research question, in natural language.")] string query,
        [Description("How many passages to return. Default 10, maximum 50.")] int k = 10,
        [Description("Restrict to one section: Methods, Results, Discussion, Introduction, Abstract, FigureCaption, TableCaption.")] string? section = null,
        [Description("Only include papers published in or after this year.")] int? yearMin = null,
        [Description("Only include papers published in or before this year.")] int? yearMax = null,
        [Description("Include retracted papers. Default true, and they are flagged.")] bool includeRetracted = true,
        [Description("Most passages to return from any single paper. Default 2.")] int maxPerArticle = 2)
    {
        if (!retrieval.HasFullText)
        {
            return JsonSerializer.Serialize(new
            {
                error = "Full text is not available on this server; only the abstracts tier is loaded.",
            }, Json);
        }

        if (string.IsNullOrWhiteSpace(query))
        {
            return "{\"error\": \"query must not be empty\"}";
        }

        var results = retrieval.SearchFullText(
            query, Math.Clamp(k, 1, 50), section, yearMin, yearMax, includeRetracted, maxPerArticle);

        return JsonSerializer.Serialize(new
        {
            query,
            section,
            returned = results.Count,
            results = results.Select(r => new
            {
                passage_id = r.Id,
                article_key = r.ArticleKey,
                title = r.Title,
                year = r.Year,
                section = string.IsNullOrEmpty(r.Section) ? null : r.Section,
                is_retracted = r.IsRetracted,
                score = Math.Round(r.Score, 3),
                text = r.Text,
            }),
        }, Json);
    }

    [McpServerTool(Name = "get_passage_context")]
    [Description(
        "Return the passages surrounding one found by search_full_text. Use when a passage " +
        "appears to start or stop mid-argument, rather than inferring what was cut off.")]
    public string GetPassageContext(
        [Description("The passage_id from search_full_text.")] string passageId,
        [Description("How many passages before it. Default 1.")] int before = 1,
        [Description("How many passages after it. Default 1.")] int after = 1)
    {
        if (!retrieval.HasFullText)
        {
            return JsonSerializer.Serialize(new { error = "Full text is not available on this server." }, Json);
        }

        var results = retrieval.GetPassageContext(passageId, Math.Clamp(before, 0, 5), Math.Clamp(after, 0, 5));
        if (results.Count == 0)
        {
            return JsonSerializer.Serialize(new { error = "not found", passage_id = passageId }, Json);
        }

        return JsonSerializer.Serialize(new
        {
            passage_id = passageId,
            passages = results.Select(r => new
            {
                passage_id = r.Id,
                article_key = r.ArticleKey,
                section = string.IsNullOrEmpty(r.Section) ? null : r.Section,
                text = r.Text,
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
            @abstract = paper.Text,
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
            abstracts = new
            {
                documents = retrieval.DocumentCount,
                use_for = "breadth - which papers exist on a topic",
            },
            full_text = retrieval.HasFullText
                ? new
                {
                    passages = retrieval.PassageCount,
                    use_for = "depth - what a study did and found",
                    coverage = "about 23% of the corpus has full text",
                    sources = "PubMed Central, bioRxiv, medRxiv, 2019-2025",
                }
                : null,
            source = "OpenAlex, filtered to any topic in field 28 (neuroscience)",
            retrieval = "BM25 (Lucene EnglishAnalyzer) then MedCPT cross-encoder reranking",
            caveats = new[]
            {
                "The field-28 filter is broad, so some papers are only tangentially neuroscience.",
                "Absence of a paper here does not mean it does not exist.",
                "Preprints and their published versions can both appear, with near-identical text.",
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
