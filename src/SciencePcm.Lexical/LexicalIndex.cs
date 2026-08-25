using Lucene.Net.Analysis;
using Lucene.Net.Analysis.En;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;
using Lucene.Net.Search.Similarities;
using Lucene.Net.Store;
using Lucene.Net.Util;

namespace SciencePcm.Lexical;

public sealed record LexicalHit(
    string Id,
    string ArticleKey,
    string Title,
    string Body,
    int Year,
    string Pmid,
    string Section,
    bool IsRetracted,
    float Score);

/// <summary>
/// BM25 over the same text the dense side embeds. Lucene replaces the DuckDB FTS
/// baseline, which ran a full scan per query at 13 s each.
///
/// Title and body are stored, not just indexed: the cross-encoder needs the passage text
/// at query time, and keeping it here avoids holding ~15 GB of abstracts in memory.
/// </summary>
public static class LexicalIndex
{
    public const LuceneVersion Version = LuceneVersion.LUCENE_48;

    public const string IdField = "id";
    public const string KeyField = "key";
    public const string BodyField = "body";
    public const string TitleField = "title";
    public const string YearField = "year";
    public const string PmidField = "pmid";
    public const string SectionField = "section";
    public const string RetractedField = "retracted";
    public const string SearchField = "body_search";

    /// <summary>Stemming and stopword removal matter more than exact term matching here.</summary>
    public static Analyzer CreateAnalyzer() => new EnglishAnalyzer(Version);

    public static Document CreateDocument(ArticleDocument source)
    {
        var document = new Document
        {
            new StringField(IdField, source.Id, Field.Store.YES),
            new StringField(KeyField, source.ArticleKey, Field.Store.YES),
            new StoredField(TitleField, source.Title),
            new StoredField(BodyField, source.Body),
            // The searchable copy carries the title, which is where much of the signal is.
            new TextField(
                SearchField,
                string.IsNullOrEmpty(source.Title) ? source.Body : source.Title + ". " + source.Body,
                Field.Store.NO),
        };

        if (source.Year > 0)
        {
            document.Add(new Int32Field(YearField, source.Year, Field.Store.YES));
        }

        if (!string.IsNullOrEmpty(source.Pmid))
        {
            document.Add(new StringField(PmidField, source.Pmid, Field.Store.YES));
        }

        if (!string.IsNullOrEmpty(source.Section))
        {
            // Lowercased and unanalysed so a section filter is an exact term match.
            document.Add(new StringField(SectionField, source.Section.ToLowerInvariant(), Field.Store.YES));
        }

        if (source.IsRetracted)
        {
            document.Add(new StringField(RetractedField, "true", Field.Store.YES));
        }

        return document;
    }

    public static string ArticleKeyOf(string id)
    {
        var hash = id.IndexOf('#');
        return hash < 0 ? id : id[..hash];
    }
}

/// <summary>What the index needs from one document, whether abstract or passage.</summary>
public sealed record ArticleDocument(
    string Id,
    string ArticleKey,
    string Title,
    string Body,
    int Year,
    string Pmid,
    string Section,
    bool IsRetracted);

public sealed class LexicalSearcher : IDisposable
{
    private readonly DirectoryReader _reader;
    private readonly FSDirectory _directory;
    private readonly IndexSearcher _searcher;
    private readonly Analyzer _analyzer;
    private readonly int _fetchMultiplier;

    public LexicalSearcher(string indexPath, int fetchMultiplier = 4)
    {
        _directory = FSDirectory.Open(indexPath);
        _reader = DirectoryReader.Open(_directory);
        _analyzer = LexicalIndex.CreateAnalyzer();
        _fetchMultiplier = Math.Max(1, fetchMultiplier);
        _searcher = new IndexSearcher(_reader) { Similarity = new BM25Similarity() };
    }

    public int Count => _reader.NumDocs;

    public List<LexicalHit> Search(
        string query,
        int k,
        int? yearMin = null,
        int? yearMax = null,
        string? section = null,
        bool includeRetracted = true)
    {
        // Built through the analyzer rather than a query parser, so that punctuation in a
        // natural-language question cannot throw a syntax error.
        var builder = new QueryBuilder(_analyzer);
        var parsed = builder.CreateBooleanQuery(LexicalIndex.SearchField, query, Occur.SHOULD);
        if (parsed is null) return [];

        Query effective = parsed;
        var needsFilter = yearMin is not null || yearMax is not null
            || !string.IsNullOrWhiteSpace(section) || !includeRetracted;

        if (needsFilter)
        {
            var combined = new BooleanQuery { { parsed, Occur.MUST } };

            if (yearMin is not null || yearMax is not null)
            {
                combined.Add(
                    NumericRangeQuery.NewInt32Range(LexicalIndex.YearField, yearMin, yearMax, true, true),
                    Occur.MUST);
            }

            if (!string.IsNullOrWhiteSpace(section))
            {
                combined.Add(
                    new TermQuery(new Term(LexicalIndex.SectionField, section.ToLowerInvariant())),
                    Occur.MUST);
            }

            if (!includeRetracted)
            {
                combined.Add(new TermQuery(new Term(LexicalIndex.RetractedField, "true")), Occur.MUST_NOT);
            }

            effective = combined;
        }

        var top = _searcher.Search(effective, k);
        var hits = new List<LexicalHit>(top.ScoreDocs.Length);

        foreach (var scoreDoc in top.ScoreDocs)
        {
            hits.Add(ToHit(_searcher.Doc(scoreDoc.Doc), scoreDoc.Score));
        }

        return hits;
    }

    private static LexicalHit ToHit(Document document, float score)
    {
        var year = document.Get(LexicalIndex.YearField);
        return new LexicalHit(
            document.Get(LexicalIndex.IdField),
            document.Get(LexicalIndex.KeyField),
            document.Get(LexicalIndex.TitleField) ?? "",
            document.Get(LexicalIndex.BodyField) ?? "",
            int.TryParse(year, out var parsed) ? parsed : 0,
            document.Get(LexicalIndex.PmidField) ?? "",
            document.Get(LexicalIndex.SectionField) ?? "",
            document.Get(LexicalIndex.RetractedField) == "true",
            score);
    }

    public List<LexicalHit> SearchArticles(string query, int k, int? yearMin = null, int? yearMax = null)
    {
        var passages = Search(query, k * _fetchMultiplier, yearMin, yearMax);
        var best = new Dictionary<string, LexicalHit>(passages.Count);

        foreach (var hit in passages)
        {
            if (!best.TryGetValue(hit.ArticleKey, out var existing) || hit.Score > existing.Score)
            {
                best[hit.ArticleKey] = hit;
            }
        }

        return best.Values.OrderByDescending(h => h.Score).Take(k).ToList();
    }

    /// <summary>Exact lookup by article key, for get_paper.</summary>
    public LexicalHit? GetByKey(string key)
    {
        var top = _searcher.Search(new TermQuery(new Term(LexicalIndex.KeyField, key)), 1);
        return top.ScoreDocs.Length == 0 ? null : ToHit(_searcher.Doc(top.ScoreDocs[0].Doc), 0f);
    }

    /// <summary>Exact lookup by document id, for walking to neighbouring passages.</summary>
    public LexicalHit? GetById(string id)
    {
        var top = _searcher.Search(new TermQuery(new Term(LexicalIndex.IdField, id)), 1);
        return top.ScoreDocs.Length == 0 ? null : ToHit(_searcher.Doc(top.ScoreDocs[0].Doc), 0f);
    }

    public void Dispose()
    {
        _reader.Dispose();
        _directory.Dispose();
        _analyzer.Dispose();
    }
}
