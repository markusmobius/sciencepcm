using Lucene.Net.Analysis;
using Lucene.Net.Analysis.En;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;
using Lucene.Net.Search.Similarities;
using Lucene.Net.Store;
using Lucene.Net.Util;

namespace SciencePcm.Lexical;

public sealed record LexicalHit(string Id, string ArticleKey, float Score);

/// <summary>
/// BM25 over the same text the dense side embeds. Lucene replaces the DuckDB FTS
/// baseline, which ran a full scan per query at 13 s each.
/// </summary>
public static class LexicalIndex
{
    public const LuceneVersion Version = LuceneVersion.LUCENE_48;

    public const string IdField = "id";
    public const string KeyField = "key";
    public const string BodyField = "body";

    /// <summary>Stemming and stopword removal matter more than exact term matching here.</summary>
    public static Analyzer CreateAnalyzer() => new EnglishAnalyzer(Version);

    public static Document CreateDocument(string id, string articleKey, string text)
    {
        var document = new Document
        {
            new StringField(IdField, id, Field.Store.YES),
            new StringField(KeyField, articleKey, Field.Store.YES),
            // Not stored: the text is already in Parquet, and storing it would double the index.
            new TextField(BodyField, text, Field.Store.NO),
        };
        return document;
    }

    public static string ArticleKeyOf(string id)
    {
        var hash = id.IndexOf('#');
        return hash < 0 ? id : id[..hash];
    }
}

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

    public List<LexicalHit> Search(string query, int k)
    {
        // Built through the analyzer rather than a query parser, so that punctuation in a
        // natural-language question cannot throw a syntax error.
        var builder = new QueryBuilder(_analyzer);
        var parsed = builder.CreateBooleanQuery(LexicalIndex.BodyField, query, Occur.SHOULD);
        if (parsed is null) return [];

        var top = _searcher.Search(parsed, k);
        var hits = new List<LexicalHit>(top.ScoreDocs.Length);

        foreach (var scoreDoc in top.ScoreDocs)
        {
            var document = _searcher.Doc(scoreDoc.Doc);
            hits.Add(new LexicalHit(
                document.Get(LexicalIndex.IdField),
                document.Get(LexicalIndex.KeyField),
                scoreDoc.Score));
        }

        return hits;
    }

    public List<LexicalHit> SearchArticles(string query, int k)
    {
        var passages = Search(query, k * _fetchMultiplier);
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

    public void Dispose()
    {
        _reader.Dispose();
        _directory.Dispose();
        _analyzer.Dispose();
    }
}
