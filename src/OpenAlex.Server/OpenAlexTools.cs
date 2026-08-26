using System.ComponentModel;
using System.Text.Json;
using ModelContextProtocol.Server;
using SciencePcm.Server;

namespace OpenAlex.Server;

[McpServerToolType]
public sealed class OpenAlexTools(RetrievalService retrieval)
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    [McpServerTool(Name = "search_openalex")]
    [Description(
        "Search abstracts across the unfiltered OpenAlex works snapshot. Use a natural-language " +
        "question and retry with alternative terminology when recall matters: BM25 produces the " +
        "candidate set before a cross-encoder reranks it. Results cover every discipline but only " +
        "works for which OpenAlex supplies an abstract.")]
    public string SearchOpenAlex(
        [Description("The research question, in natural language.")] string query,
        [Description("How many works to return. Default 10, maximum 50.")] int k = 10,
        [Description("Only include works published in or after this year.")] int? yearMin = null,
        [Description("Only include works published in or before this year.")] int? yearMax = null,
        [Description("Skip cross-encoder reranking. Faster but less relevant.")] bool fast = false)
    {
        if (string.IsNullOrWhiteSpace(query)) return "{\"error\": \"query must not be empty\"}";

        var results = retrieval.Search(query, Math.Clamp(k, 1, 50), yearMin, yearMax, rerank: !fast);
        return JsonSerializer.Serialize(new
        {
            query,
            returned = results.Count,
            results = results.Select(result => new
            {
                openalex_id = result.ArticleKey,
                title = result.Title,
                year = result.Year,
                pmid = string.IsNullOrEmpty(result.Pmid) ? null : result.Pmid,
                score = Math.Round(result.Score, 3),
                abstract_excerpt = Excerpt(result.Text, 500),
            }),
        }, Json);
    }

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
            year = work.Year,
            pmid = string.IsNullOrEmpty(work.Pmid) ? null : work.Pmid,
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
}