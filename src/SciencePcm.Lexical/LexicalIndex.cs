using Lucene.Net.Analysis;
using Lucene.Net.Analysis.En;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Queries;
using Lucene.Net.Search;
using Lucene.Net.Search.Similarities;
using Lucene.Net.Store;
using Lucene.Net.Util;
using SciencePcm.Embed;

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
    float Score,
    BibliographicMetadata? Metadata = null);

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
    public const string PublicationDateField = "publication_date";
    public const string DoiField = "doi";
    public const string AuthorsField = "authors";
    public const string InstitutionsField = "institutions";
    public const string JournalField = "journal";
    public const string IssnField = "issn";
    public const string LanguageField = "language";
    public const string WorkTypeField = "work_type";
    public const string CitedByCountField = "cited_by_count";
    public const string VolumeField = "volume";
    public const string IssueField = "issue";
    public const string FirstPageField = "first_page";
    public const string LastPageField = "last_page";
    public const string TopicsField = "topics";
    public const string KeywordsField = "keywords";
    public const string PmcidField = "pmcid";
    public const string PublisherField = "publisher";
    public const string LandingPageUrlField = "landing_page_url";
    public const string PdfUrlField = "pdf_url";
    public const string LicenseField = "license";
    public const string OpenAccessField = "open_access";
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
            // Bibliographic names are candidate anchors when matching prose that mentions
            // researchers, institutions or venues but does not quote the abstract.
            new TextField(
                SearchField,
                SearchableText(source),
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

        if (source.Metadata is { } metadata)
        {
            AddStored(document, PublicationDateField, metadata.PublicationDate);
            AddStored(document, DoiField, metadata.Doi);
            AddStored(document, AuthorsField, metadata.Authors);
            AddStored(document, InstitutionsField, metadata.Institutions);
            AddStored(document, JournalField, metadata.Journal);
            AddStored(document, IssnField, metadata.Issn);
            AddStored(document, LanguageField, metadata.Language);
            if (!string.IsNullOrEmpty(metadata.WorkType))
            {
                // Indexed, not merely stored, so peer reviews and datasets - which OpenAlex
                // records alongside the paper they discuss - can be excluded by query.
                document.Add(new StringField(
                    WorkTypeField, metadata.WorkType.ToLowerInvariant(), Field.Store.YES));
            }
            AddStored(document, VolumeField, metadata.Volume);
            AddStored(document, IssueField, metadata.Issue);
            AddStored(document, FirstPageField, metadata.FirstPage);
            AddStored(document, LastPageField, metadata.LastPage);
            AddStored(document, TopicsField, metadata.Topics);
            AddStored(document, KeywordsField, metadata.Keywords);
            AddStored(document, PmcidField, metadata.Pmcid);
            AddStored(document, PublisherField, metadata.Publisher);
            AddStored(document, LandingPageUrlField, metadata.LandingPageUrl);
            AddStored(document, PdfUrlField, metadata.PdfUrl);
            AddStored(document, LicenseField, metadata.License);
            if (metadata.IsOpenAccess is { } isOpenAccess)
            {
                document.Add(new StringField(OpenAccessField, isOpenAccess ? "true" : "false", Field.Store.YES));
            }
            if (metadata.CitedByCount is { } citedByCount)
            {
                document.Add(new Int32Field(CitedByCountField, citedByCount, Field.Store.YES));
            }
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

    private static string SearchableText(ArticleDocument source)
    {
        var metadata = source.Metadata;
        return string.Join(". ", new[]
        {
            source.Title,
            source.Body,
            // Capped: a 765-author trial paper carries 13,000 characters of names, which
            // is 80% of its searchable text, and BM25 divides every term's contribution
            // by document length. Uncapped, collaboration papers sank below the news
            // pieces and errata written about them. News prose names the first authors.
            Truncate(metadata?.Authors, 400),
            Truncate(metadata?.Institutions, 400),
            metadata?.Journal,
            metadata?.Doi,
            metadata?.Pmcid,
            metadata?.Issn,
            metadata?.Publisher,
            metadata?.Topics,
            metadata?.Keywords,
        }.Where(value => !string.IsNullOrWhiteSpace(value)));
    }

    private static string? Truncate(string? value, int length)
    {
        if (string.IsNullOrEmpty(value) || value.Length <= length) return value;
        var cut = value.LastIndexOf(';', length);
        return value[..(cut > 0 ? cut : length)];
    }

    private static void AddStored(Document document, string field, string value)
    {
        if (!string.IsNullOrEmpty(value)) document.Add(new StoredField(field, value));
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
    bool IsRetracted,
    BibliographicMetadata? Metadata = null);

/// <summary>
/// Tilts the BM25 score towards well-cited work, by a bounded multiplicative factor.
/// </summary>
/// <remarks>
/// News prose names a paper, but the corpus also holds the reviews, errata, preprints
/// and duplicate records that quote its title, and those are shorter, so BM25 prefers
/// them. Citations separate the two populations by orders of magnitude.
///
/// Multiplicative and capped rather than additive: an additive bonus has to be tuned
/// against BM25's score scale, and at any weight large enough to matter it lets the
/// most-cited papers in the corpus win every query outright. Scaling by at most
/// (1 + weight) means citations break ties between topically similar documents but can
/// never promote an irrelevant one.
/// </remarks>
public sealed class CitationBoostQuery(Query subQuery, double weight) : CustomScoreQuery(subQuery)
{
    // log10 of the most-cited work in OpenAlex, near enough: ~180k citations.
    private const double Ceiling = 5.5;

    private readonly double _weight = weight;

    public override string Name => "citationBoost";

    protected override CustomScoreProvider GetCustomScoreProvider(AtomicReaderContext context) =>
        new Provider(context, _weight);

    private sealed class Provider(AtomicReaderContext context, double weight)
        : CustomScoreProvider(context)
    {
        // Int32Field is trie-encoded, so the parser has to be named explicitly: the
        // parserless overload reads zero for every document and the boost silently
        // does nothing.
        private readonly FieldCache.Int32s _citations = FieldCache.DEFAULT.GetInt32s(
            context.AtomicReader, LexicalIndex.CitedByCountField,
            FieldCache.NUMERIC_UTILS_INT32_PARSER, true);

        public override float CustomScore(int doc, float subQueryScore, float valSrcScore)
        {
            var citations = _citations.Get(doc);
            if (citations <= 0) return subQueryScore;

            var share = Math.Min(1.0, Math.Log10(1 + citations) / Ceiling);
            return (float)(subQueryScore * (1 + weight * share));
        }

        public override float CustomScore(int doc, float subQueryScore, float[] valSrcScores) =>
            CustomScore(doc, subQueryScore, 0f);
    }
}

public sealed class LexicalSearcher : IDisposable
{
    private readonly DirectoryReader _reader;
    private readonly FSDirectory _directory;
    private readonly IndexSearcher _searcher;
    private readonly Analyzer _analyzer;
    private readonly int _fetchMultiplier;
    private readonly double _maxDocFreqRatio;
    private readonly double _citationPriorWeight;

    /// <param name="parallel">
    /// Search index segments concurrently. On a 256M document index a long query spends
    /// seconds scoring on one core while the rest of the machine idles.
    /// </param>
    /// <param name="maxDocFreqRatio">
    /// Drop query terms appearing in more than this fraction of documents. Words like
    /// "researchers" or "reported" match tens of millions of documents and contribute
    /// almost nothing to ranking, but Lucene 4.8 has no BlockMax-WAND so it scores every
    /// one of them. 0 disables the filter.
    /// </param>
    public LexicalSearcher(
        string indexPath,
        int fetchMultiplier = 4,
        bool parallel = true,
        double maxDocFreqRatio = 0,
        double citationPriorWeight = 0,
        float bm25B = 0.75f)
    {
        _directory = FSDirectory.Open(indexPath);
        _reader = DirectoryReader.Open(_directory);
        _analyzer = LexicalIndex.CreateAnalyzer();
        _fetchMultiplier = Math.Max(1, fetchMultiplier);
        _maxDocFreqRatio = maxDocFreqRatio;
        _citationPriorWeight = citationPriorWeight;

        _searcher = parallel
            ? new IndexSearcher(_reader, TaskScheduler.Default)
            : new IndexSearcher(_reader);

        // b controls length normalisation, and is read from the stored norm at query
        // time, so it is tunable without reindexing. Papers by large collaborations
        // carry hundreds of author names in the searched field, which the default 0.75
        // penalises heavily.
        _searcher.Similarity = new BM25Similarity(1.2f, bm25B);
    }

    public int Count => _reader.NumDocs;

    /// <summary>
    /// Distinct article keys, which for the passage index is how many papers have full
    /// text - not the same as the passage count, since one paper yields many passages.
    /// Returns -1 when the codec cannot count terms without a full scan.
    /// </summary>
    public long DistinctArticleCount
    {
        get
        {
            var terms = MultiFields.GetTerms(_reader, LexicalIndex.KeyField);
            return terms?.Count ?? -1;
        }
    }

    public List<LexicalHit> Search(
        string query,
        int k,
        int? yearMin = null,
        int? yearMax = null,
        string? section = null,
        bool includeRetracted = true,
        IReadOnlyCollection<string>? excludeWorkTypes = null)
    {
        // Built through the analyzer rather than a query parser, so that punctuation in a
        // natural-language question cannot throw a syntax error.
        var builder = new QueryBuilder(_analyzer);
        var parsed = builder.CreateBooleanQuery(LexicalIndex.SearchField, query, Occur.SHOULD);
        if (parsed is null) return [];

        parsed = TrimCommonTerms(parsed);
        if (parsed is null) return [];

        Query effective = parsed;
        var needsFilter = yearMin is not null || yearMax is not null
            || !string.IsNullOrWhiteSpace(section) || !includeRetracted
            || excludeWorkTypes is { Count: > 0 };

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

            if (excludeWorkTypes is { Count: > 0 })
            {
                foreach (var workType in excludeWorkTypes)
                {
                    if (string.IsNullOrWhiteSpace(workType)) continue;
                    combined.Add(
                        new TermQuery(new Term(LexicalIndex.WorkTypeField, workType.ToLowerInvariant())),
                        Occur.MUST_NOT);
                }
            }

            effective = combined;
        }

        if (_citationPriorWeight > 0)
        {
            effective = new CitationBoostQuery(effective, _citationPriorWeight);
        }

        var top = _searcher.Search(effective, k);
        var hits = new List<LexicalHit>(top.ScoreDocs.Length);

        foreach (var scoreDoc in top.ScoreDocs)
        {
            hits.Add(ToHit(_searcher.Doc(scoreDoc.Doc), scoreDoc.Score));
        }

        return hits;
    }

    /// <summary>
    /// Removes terms so common that scoring them costs far more than they inform. Keeps
    /// everything if that would leave nothing to search on - a query made entirely of
    /// common words still has to return something.
    /// </summary>
    private BooleanQuery? TrimCommonTerms(Query query)
    {
        if (_maxDocFreqRatio <= 0 || query is not BooleanQuery boolean) return query as BooleanQuery;

        var ceiling = (int)(_reader.NumDocs * _maxDocFreqRatio);
        var kept = new BooleanQuery();
        var dropped = 0;

        foreach (var clause in boolean.Clauses)
        {
            if (clause.Query is TermQuery term && _reader.DocFreq(term.Term) > ceiling)
            {
                dropped++;
                continue;
            }
            kept.Add(clause);
        }

        if (kept.Clauses.Count == 0) return boolean;
        return dropped == 0 ? boolean : kept;
    }

    private static LexicalHit ToHit(Document document, float score)
    {
        var year = document.Get(LexicalIndex.YearField);
        var citedByCount = document.Get(LexicalIndex.CitedByCountField);
        var metadata = new BibliographicMetadata(
            document.Get(LexicalIndex.PublicationDateField) ?? "",
            document.Get(LexicalIndex.DoiField) ?? "",
            document.Get(LexicalIndex.AuthorsField) ?? "",
            document.Get(LexicalIndex.InstitutionsField) ?? "",
            document.Get(LexicalIndex.JournalField) ?? "",
            document.Get(LexicalIndex.IssnField) ?? "",
            document.Get(LexicalIndex.LanguageField) ?? "",
            document.Get(LexicalIndex.WorkTypeField) ?? "",
            int.TryParse(citedByCount, out var citations) ? citations : null,
            document.Get(LexicalIndex.VolumeField) ?? "",
            document.Get(LexicalIndex.IssueField) ?? "",
            document.Get(LexicalIndex.FirstPageField) ?? "",
            document.Get(LexicalIndex.LastPageField) ?? "",
            document.Get(LexicalIndex.TopicsField) ?? "",
            document.Get(LexicalIndex.KeywordsField) ?? "",
            document.Get(LexicalIndex.PmcidField) ?? "",
            document.Get(LexicalIndex.PublisherField) ?? "",
            document.Get(LexicalIndex.LandingPageUrlField) ?? "",
            document.Get(LexicalIndex.PdfUrlField) ?? "",
            document.Get(LexicalIndex.LicenseField) ?? "",
            document.Get(LexicalIndex.OpenAccessField) switch
            {
                "true" => true,
                "false" => false,
                _ => null,
            });
        return new LexicalHit(
            document.Get(LexicalIndex.IdField),
            document.Get(LexicalIndex.KeyField),
            document.Get(LexicalIndex.TitleField) ?? "",
            document.Get(LexicalIndex.BodyField) ?? "",
            int.TryParse(year, out var parsed) ? parsed : 0,
            document.Get(LexicalIndex.PmidField) ?? "",
            document.Get(LexicalIndex.SectionField) ?? "",
            document.Get(LexicalIndex.RetractedField) == "true",
            score,
            metadata);
    }

    public List<LexicalHit> SearchArticles(
        string query,
        int k,
        int? yearMin = null,
        int? yearMax = null,
        IReadOnlyCollection<string>? excludeWorkTypes = null)
    {
        var passages = Search(
            query, k * _fetchMultiplier, yearMin, yearMax, excludeWorkTypes: excludeWorkTypes);
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

    /// <summary>
    /// Lucene's own account of a score: which clauses matched, their IDF and term
    /// frequency, and the length norm. The only way to tell "this document scores low"
    /// apart from "this document never matched" without guessing.
    /// </summary>
    public string Explain(string query, string id)
    {
        var top = _searcher.Search(new TermQuery(new Term(LexicalIndex.IdField, id)), 1);
        if (top.ScoreDocs.Length == 0) return $"'{id}' is not in this index.";

        var builder = new QueryBuilder(_analyzer);
        var parsed = builder.CreateBooleanQuery(LexicalIndex.SearchField, query, Occur.SHOULD);
        if (parsed is null) return "Query produced no terms.";

        Query effective = TrimCommonTerms(parsed) ?? parsed;
        if (_citationPriorWeight > 0)
        {
            effective = new CitationBoostQuery(effective, _citationPriorWeight);
        }

        return _searcher.Explain(effective, top.ScoreDocs[0].Doc).ToString();
    }

    /// <summary>Analyzed terms of a query, with how many documents each matches.</summary>
    public IEnumerable<(string Term, int DocFreq, bool Kept)> QueryTerms(string query)
    {
        var builder = new QueryBuilder(_analyzer);
        if (builder.CreateBooleanQuery(LexicalIndex.SearchField, query, Occur.SHOULD)
            is not BooleanQuery parsed) yield break;

        var ceiling = _maxDocFreqRatio > 0 ? (int)(_reader.NumDocs * _maxDocFreqRatio) : int.MaxValue;

        foreach (var clause in parsed.Clauses)
        {
            if (clause.Query is not TermQuery term) continue;
            var frequency = _reader.DocFreq(term.Term);
            yield return (term.Term.Text, frequency, frequency <= ceiling);
        }
    }

    public void Dispose()
    {
        _reader.Dispose();
        _directory.Dispose();
        _analyzer.Dispose();
    }
}
