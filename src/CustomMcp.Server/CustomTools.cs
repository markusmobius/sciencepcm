using System.ComponentModel;
using System.Text.Json;
using ModelContextProtocol.Server;
using SciencePcm.Server;

namespace CustomMcp.Server;

[McpServerToolType]
public sealed class CustomTools(CustomCollection collection)
{
    private readonly LiteratureTools _literature = new(collection.Retrieval);

    [McpServerTool(Name = "search_literature")]
    [Description("Search titles and abstracts only in this endpoint's named PDS collection. Returns paper keys, authors, DOI, journal, " +
        "and abstract excerpts. Use a natural-language research question; retry with alternate technical terminology when needed. " +
        "Author and journal filters can be used without a query. Use search_full_text for methods and findings, " +
        "and get_paper for the complete abstract. Missing dates and citation counts are unknown; year filters exclude undated papers.")]
    public string SearchLiterature(
        [Description("Research question; optional with an author or journal filter.")] string query = "",
        [Description("Number of papers, default 10, maximum 50.")] int limit = 10,
        [Description("Author name, preferably surname. Supports OR and AND.")] string author = "",
        [Description("Journal phrase; supports OR.")] string journal = "",
        [Description("relevance, citations, or year. Source dates and citations may be absent.")] string sort = "",
        [Description("Earliest publication year; excludes undated papers.")] int? yearMin = null,
        [Description("Latest publication year; excludes undated papers.")] int? yearMax = null,
        [Description("Skip cross-encoder reranking; default false.")] bool fast = false) =>
        _literature.SearchLiterature(query, limit, author, journal, sort, yearMin, yearMax, fast);

    [McpServerTool(Name = "search_full_text")]
    [Description("Search full-text passages only in this endpoint's named PDS collection for methods, parameters, procedures, and findings. " +
        "Results include passage IDs, paper keys, authors, DOI, section, and text. Use get_passage_context to expand a hit. " +
        "Absence of a result in this bounded collection is not evidence that a paper does not exist.")]
    public string SearchFullText(
        [Description("Research question in natural language.")] string query,
        [Description("Number of passages, default 10, maximum 50.")] int limit = 10,
        [Description("Section such as Methods, Results, Discussion, Introduction, Abstract, FigureCaption, or TableCaption.")] string section = "",
        [Description("Earliest publication year; excludes undated papers.")] int? yearMin = null,
        [Description("Latest publication year; excludes undated papers.")] int? yearMax = null,
        [Description("Include records marked retracted, default true.")] bool includeRetracted = true,
        [Description("Maximum passages per paper, default 2.")] int maxPerArticle = 2) =>
        _literature.SearchFullText(query, limit, section, yearMin, yearMax, includeRetracted, maxPerArticle);

    [McpServerTool(Name = "get_paper")]
    [Description("Return the complete abstract and available bibliographic metadata for a paper in this endpoint's collection. " +
        "The source may omit dates or an abstract. This does not return full text or access another collection.")]
    public string GetPaper([Description("Exact <collection>:<PDS key> returned by this endpoint's search.")] string articleKey) =>
        _literature.GetPaper(articleKey);

    [McpServerTool(Name = "get_passage_context")]
    [Description("Return passages surrounding a full-text hit in this endpoint's collection, preserving their order within the paper.")]
    public string GetPassageContext(
        [Description("Exact passage_id returned by this endpoint's search_full_text.")] string passageId,
        [Description("Passages before the hit, default 1, maximum 5.")] int before = 1,
        [Description("Passages after the hit, default 1, maximum 5.")] int after = 1) =>
        _literature.GetPassageContext(passageId, before, after);

    [McpServerTool(Name = "corpus_stats")]
    [Description("Return the collection name, source, measured index counts, and limitations for this endpoint only.")]
    public string CorpusStats() => JsonSerializer.Serialize(new
    {
        service = "CustomMcp",
        corpus = collection.Definition.Name,
        documents = collection.Retrieval.DocumentCount,
        passages = collection.Retrieval.PassageCount,
        papers_with_passages = collection.Retrieval.FullTextArticleCount,
        source = collection.Definition.InputCloudPath,
        source_sha256 = collection.Prepared.InputSha256,
        retrieval = "Fielded Lucene BM25 followed by a shared ONNX cross-encoder",
        caveats = new[]
        {
            "Only this configured PDS collection is searched, not other CustomMcp endpoints or the wider literature.",
            "Document counts include title-only records; not every record has a published abstract.",
            "Missing publication dates remain unknown (year 0); year filters exclude them.",
            "Citation counts and open-access status are not inferred from the presence of full text.",
            "Text and mathematical notation may contain source extraction artifacts.",
        },
    });
}