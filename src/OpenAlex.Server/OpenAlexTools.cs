using System.ComponentModel;
using System.Text.Json;
using ModelContextProtocol.Server;
using SciencePcm.Index;
using SciencePcm.Server;

namespace OpenAlex.Server;

[McpServerToolType]
public sealed class OpenAlexTools(RetrievalService retrieval)
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    [McpServerTool(Name = "search_openalex")]
    [Description(
        "Find papers across the unfiltered OpenAlex works snapshot. The query may be a research " +
        "question, a claim or passage from a newspaper article, or remembered bibliographic clues. " +
        "Title, abstract, authors, institutions, journal, identifiers and topics are separate " +
        "indexed fields, matched together and weighted, before a cross-encoder reranks candidates. " +
        "Set author or journal to restrict rather than merely favour: both match the stored name as " +
        "a phrase, so a surname alone is the reliable way to list a researcher's papers. " +
        "With author or journal set, query may be omitted, and sort=citations or sort=year is " +
        "usually what you want, since relevance means little when browsing. " +
        "Works without abstracts are included and are about a fifth of recent articles.")]
    public string SearchOpenAlex(
        [Description("A question, news passage, claim, title fragment, institution or identifier. Optional when author or journal is set.")] string query = "",
        [Description("How many works to return. Default 10, maximum 50.")] int limit = 10,
        [Description("Author name. A surname alone is most reliable, for example 'Rosenblat'; " +
            "either 'Tanya Rosenblat' or 'Rosenblat, Tanya' also works, but a fuller name only " +
            "matches records that spell it that way. Retry with the surname alone if empty.")] string author = "",
        [Description("Journal name, matched as a phrase, so 'Lancet' finds 'The Lancet'. " +
            "One journal at a time.")] string journal = "",
        [Description("Order: relevance (default), citations, or year.")] string sort = "",
        [Description("Only include works published in or after this year.")] int? yearMin = null,
        [Description("Only include works published in or before this year.")] int? yearMax = null,
        [Description("Skip cross-encoder reranking. Faster but less relevant.")] bool fast = false)
    {
        var hasFilter = !string.IsNullOrWhiteSpace(author) || !string.IsNullOrWhiteSpace(journal);
        if (string.IsNullOrWhiteSpace(query) && !hasFilter)
        {
            return "{\"error\": \"pass query, or author, or journal\"}";
        }

        if (!SortOrders.TryParse(sort, out var order))
        {
            return "{\"error\": \"sort must be relevance, citations or year\"}";
        }

        var results = retrieval.Search(
            query, Math.Clamp(limit, 1, 50), yearMin, yearMax, rerank: !fast,
            author: author, journal: journal, sort: order);

        return JsonSerializer.Serialize(new
        {
            query,
            author,
            journal,
            sort = order.ToString().ToLowerInvariant(),
            returned = results.Count,
            results = results.Select(result => new
            {
                openalex_id = result.ArticleKey,
                title = result.Title,
                authors = Value(result.Metadata?.Authors),
                institutions = Value(result.Metadata?.Institutions),
                journal = Value(result.Metadata?.Journal),
                publication_date = Value(result.Metadata?.PublicationDate),
                year = result.Year,
                doi = Value(result.Metadata?.Doi),
                pmid = string.IsNullOrEmpty(result.Pmid) ? null : result.Pmid,
                type = Value(result.Metadata?.WorkType),
                language = Value(result.Metadata?.Language),
                cited_by_count = result.Metadata?.CitedByCount,
                topics = Value(result.Metadata?.Topics),
                is_retracted = result.IsRetracted,
                score = Math.Round(result.Score, 3),
                abstract_excerpt = Excerpt(result.Text, 500),
            }),
        }, Json);
    }

    private static bool TryParseSort(string? value, out SortOrder order) =>
        SortOrders.TryParse(value, out order);

    [McpServerTool(Name = "get_openalex_work")]
    [Description("Retrieve the complete abstract for one work returned by search_openalex.")]
    public string GetOpenAlexWork(
        [Description("The OpenAlex ID, for example https://openalex.org/W2154021234.")] string openAlexId)
    {
        var work = retrieval.GetPaper(openAlexId);
        if (work is null)
        {
            return JsonSerializer.Serialize(new { error = "not found", openalex_id = openAlexId }, Json);
        }

        return JsonSerializer.Serialize(new
        {
            openalex_id = work.ArticleKey,
            title = work.Title,
            authors = Value(work.Metadata?.Authors),
            institutions = Value(work.Metadata?.Institutions),
            journal = Value(work.Metadata?.Journal),
            issn = Value(work.Metadata?.Issn),
            publication_date = Value(work.Metadata?.PublicationDate),
            year = work.Year,
            doi = Value(work.Metadata?.Doi),
            pmid = string.IsNullOrEmpty(work.Pmid) ? null : work.Pmid,
            type = Value(work.Metadata?.WorkType),
            language = Value(work.Metadata?.Language),
            cited_by_count = work.Metadata?.CitedByCount,
            volume = Value(work.Metadata?.Volume),
            issue = Value(work.Metadata?.Issue),
            first_page = Value(work.Metadata?.FirstPage),
            last_page = Value(work.Metadata?.LastPage),
            topics = Value(work.Metadata?.Topics),
            keywords = Value(work.Metadata?.Keywords),
            is_retracted = work.IsRetracted,
            @abstract = work.Text,
        }, Json);
    }

    [McpServerTool(Name = "openalex_corpus_stats")]
    [Description("Describe the OpenAlex corpus and retrieval limitations before interpreting absent results.")]
    public string CorpusStats() => JsonSerializer.Serialize(new
    {
        service = "OpenAlex abstracts",
        documents = retrieval.DocumentCount,
        source = "Complete local OpenAlex works snapshot, without a field or topic filter",
        inclusion = "Works with a nonempty abstract_inverted_index",
        retrieval = "Lucene BM25 candidate retrieval followed by cross-encoder reranking",
        caveats = new[]
        {
            "OpenAlex does not provide an abstract for every work; missing works are not searchable here.",
            "Lexical candidate retrieval can miss papers that use different terminology.",
            "English analysis and reranking are weaker for non-English abstracts.",
            "OpenAlex can contain duplicate records or multiple versions of substantially the same work.",
        },
    }, Json);

    private static string Excerpt(string text, int limit)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= limit) return text;
        var cut = text.LastIndexOf(' ', Math.Min(limit, text.Length - 1));
        return text[..(cut > 0 ? cut : limit)] + " ...";
    }

    private static string? Value(string? value) => string.IsNullOrEmpty(value) ? null : value;
}