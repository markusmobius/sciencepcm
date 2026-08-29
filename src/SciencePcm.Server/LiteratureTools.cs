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
        "Each result includes title, journal, publication date/year, DOI, PMID, PMCID, citation " +
        "count, open-access status, relevance score and an abstract excerpt when available. " +
        "Use get_paper for the complete abstract and additional links. " +
        "DO NOT use abstracts to answer methods questions, sample sizes, parameters, procedures " +
        "or specific findings: call search_full_text instead. On a methods benchmark, full-text " +
        "search achieved 0.932 nDCG@10 and 89% useful results, versus 0.247 and 21% for abstracts. " +
        "Use a full natural-language question rather than keywords; the reranker reads the " +
        "question and the abstract together, so phrasing carries information. " +
        "Candidates are found lexically before being reranked, so a paper that uses different " +
        "terminology to your question may not be found at all: if results look thin or you " +
        "suspect other wording exists (a gene symbol vs its full name, 'chromosome' vs " +
        "'genetics'), CALL THIS AGAIN with the alternative terms and merge what you get. " +
        "Set author or journal to restrict rather than merely favour: author expects the form " +
        "'Surname, Given' and matches exactly, so use it to list a researcher's papers. With " +
        "author or journal set, query may be omitted, and sort=citations or sort=year is " +
        "usually what you want, since relevance means little when browsing.")]
    public string SearchLiterature(
        [Description("The research question, in natural language. Optional when author or journal is set.")] string query = "",
        [Description("How many papers to return. Default 10, maximum 50.")] int limit = 10,
        [Description("Exact author, for example 'Doudna, Jennifer'.")] string author = "",
        [Description("Exact journal name, for example 'Nature Neuroscience'.")] string journal = "",
        [Description("Order: relevance (default), citations, or year.")] string sort = "",
        [Description("Only include papers published in or after this year.")] int? yearMin = null,
        [Description("Only include papers published in or before this year.")] int? yearMax = null,
        [Description("Skip reranking. Faster, noticeably less relevant. Default false.")] bool fast = false)
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
            results = results.Select(r => new
            {
                article_key = r.ArticleKey,
                title = r.Title,
                authors = Value(r.Metadata?.Authors),
                journal = Value(r.Metadata?.Journal),
                publication_date = Value(r.Metadata?.PublicationDate),
                year = r.Year,
                doi = Value(r.Metadata?.Doi),
                pmid = string.IsNullOrEmpty(r.Pmid) ? null : r.Pmid,
                pmcid = Value(r.Metadata?.Pmcid),
                cited_by_count = r.Metadata?.CitedByCount,
                is_open_access = r.Metadata?.IsOpenAccess,
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
        "papers. ALWAYS use this first for what a study actually did or found: methods, sample " +
        "sizes, parameters, procedures and specific results. Full text substantially outperformed " +
        "abstracts on methods questions (0.932 vs 0.247 nDCG@10; 89% vs 21% useful results). " +
        "Each hit is a ~300 word fragment with title, authors, journal, publication date/year, " +
        "DOI, PMID, PMCID, section and retraction status, so it can be quoted and attributed. " +
        "IMPORTANT: most papers in this corpus have NO full text, and full text is 2019-2025 " +
        "only, so absence here does not mean absence from the literature. Call corpus_stats for " +
        "the measured coverage. Fall back to search_literature to identify papers, " +
        "but do not present an abstract as evidence for methodological details it does not state. " +
        "Retracted papers ARE included and flagged is_retracted; say so if you cite one. " +
        "As with search_literature, candidates are found lexically, so re-query with the " +
        "terminology a paper would actually use rather than the phrasing of the question.")]
    public string SearchFullText(
        [Description("The research question, in natural language.")] string query,
        [Description("How many passages to return. Default 10, maximum 50.")] int limit = 10,
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
            query, Math.Clamp(limit, 1, 50), section, yearMin, yearMax, includeRetracted, maxPerArticle);

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
                authors = Value(r.Metadata?.Authors),
                journal = Value(r.Metadata?.Journal),
                publication_date = Value(r.Metadata?.PublicationDate),
                year = r.Year,
                doi = Value(r.Metadata?.Doi),
                pmid = string.IsNullOrEmpty(r.Pmid) ? null : r.Pmid,
                pmcid = Value(r.Metadata?.Pmcid),
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
    [Description(
        "Retrieve the complete abstract and all available bibliographic metadata for one paper " +
        "by its article key: title, journal, publisher, publication date/year, DOI, PMID, PMCID, " +
        "ISSN, work type, language, citation count, keywords, open-access status, landing/PDF URLs, " +
        "license and retraction status. Fields absent from the source are null. This does not " +
        "return article full text; use search_full_text for passages.")]
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
            authors = Value(paper.Metadata?.Authors),
            journal = Value(paper.Metadata?.Journal),
            publisher = Value(paper.Metadata?.Publisher),
            publication_date = Value(paper.Metadata?.PublicationDate),
            year = paper.Year,
            doi = Value(paper.Metadata?.Doi),
            pmid = string.IsNullOrEmpty(paper.Pmid) ? null : paper.Pmid,
            pmcid = Value(paper.Metadata?.Pmcid),
            issn = Value(paper.Metadata?.Issn),
            type = Value(paper.Metadata?.WorkType),
            language = Value(paper.Metadata?.Language),
            cited_by_count = paper.Metadata?.CitedByCount,
            keywords = Value(paper.Metadata?.Keywords),
            is_open_access = paper.Metadata?.IsOpenAccess,
            landing_page_url = Value(paper.Metadata?.LandingPageUrl),
            pdf_url = Value(paper.Metadata?.PdfUrl),
            license = Value(paper.Metadata?.License),
            is_retracted = paper.IsRetracted,
            @abstract = paper.Text,
        }, Json);
    }

    [McpServerTool(Name = "corpus_stats")]
    [Description(
        "Describe what this corpus covers and how it was built. Call this before drawing " +
        "conclusions from an absence of results.")]
    public string CorpusStats()
    {
        // Measured from the passage index rather than remembered: distinct article keys,
        // not passage count, is how many papers actually have full text.
        var fullTextPapers = retrieval.FullTextArticleCount;
        var coverage = fullTextPapers > 0 && retrieval.DocumentCount > 0
            ? $"{100.0 * fullTextPapers / retrieval.DocumentCount:F1}% of indexed papers have full text"
            : "unknown";

        return JsonSerializer.Serialize(new
        {
            abstracts = new
            {
                documents = retrieval.DocumentCount,
                use_for = "breadth - which papers exist on a topic",
                not_for = "methods, sample sizes, parameters, procedures or specific findings",
            },
            full_text = retrieval.HasFullText
                ? new
                {
                    passages = retrieval.PassageCount,
                    papers = fullTextPapers,
                    use_for = "methods and depth - what a study did and found",
                    coverage,
                    sources = "PubMed Central, bioRxiv, medRxiv, 2019-2025",
                    methods_benchmark = new
                    {
                        ndcg_at_10 = 0.9317,
                        mean_grade = 2.603,
                        useful_result_percent = 89.0,
                        abstract_ndcg_at_10 = 0.2472,
                        abstract_useful_result_percent = 20.7,
                    },
                }
                : null,
            source = "OpenAlex, filtered to any topic in field 28 (neuroscience)",
            retrieval = "BM25 (Lucene EnglishAnalyzer) then BGE-reranker-v2-m3 cross-encoder reranking",
            metadata = new
            {
                abstracts = "DOI, PMID, PMCID, publication date, journal, citations, language, type and open-access links",
                full_text = "DOI, PMID, PMCID, authors, publication date, journal, publisher, ISSN, type, keywords and license",
            },
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

    private static string? Value(string? value) => string.IsNullOrEmpty(value) ? null : value;
}
