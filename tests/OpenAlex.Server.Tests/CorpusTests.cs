using Lucene.Net.Index;
using Lucene.Net.Store;
using System.Text.Json;
using SciencePcm.Embed;
using SciencePcm.Index;
using SciencePcm.Server;
using Xunit;

namespace OpenAlex.Server.Tests;

public sealed class CorpusTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "openalex-tests-" + Guid.NewGuid());
    private readonly LexicalSearcher _searcher;
    private static readonly string[] AllowedKeys = ["https://openalex.org/W9001", "https://openalex.org/W9002"];

    public CorpusTests()
    {
        System.IO.Directory.CreateDirectory(_directory);
        using (var directory = FSDirectory.Open(_directory))
        using (var analyzer = LexicalIndex.CreateAnalyzer())
        using (var writer = new IndexWriter(directory, new IndexWriterConfig(LexicalIndex.Version, analyzer)))
        {
            for (var index = 1; index <= 80; index++)
                AddWork(writer, $"https://openalex.org/W{index}", $"quartz {index}", "quartz", 2025, 1000 + index);

            AddWork(writer, AllowedKeys[0], "First subset work", "quartz", 2019, 10);
            AddWork(writer, AllowedKeys[1], "Second subset work", "quartz", 2020, 20);
        }

        _searcher = new LexicalSearcher(_directory);
    }

    [Theory]
    [InlineData(SortOrder.Relevance)]
    [InlineData(SortOrder.Citations)]
    [InlineData(SortOrder.Year)]
    public void FilterIsAppliedBeforeTopCandidates(SortOrder sort)
    {
        var filter = LexicalIndex.CreateKeyFilter(AllowedKeys);
        var unrestricted = _searcher.SearchArticles("quartz", 20, sort: sort);
        Assert.DoesNotContain(unrestricted, hit => AllowedKeys.Contains(hit.ArticleKey));

        var restricted = _searcher.SearchArticles("quartz", 20, sort: sort, corpusFilter: filter);
        Assert.Equal(AllowedKeys.Order(), restricted.Select(hit => hit.ArticleKey).Order());
    }

    [Fact]
    public void FilterDoesNotChangeRelevanceScores()
    {
        var filter = LexicalIndex.CreateKeyFilter(AllowedKeys);
        var unrestricted = _searcher.Search("quartz", 100).ToDictionary(hit => hit.ArticleKey);
        var restricted = _searcher.Search("quartz", 20, corpusFilter: filter);
        Assert.All(restricted, hit => Assert.Equal(unrestricted[hit.ArticleKey].Score, hit.Score));
    }

    [Fact]
    public void FilterCombinesWithAuthorJournalAndYear()
    {
        var filter = LexicalIndex.CreateKeyFilter(AllowedKeys);
        var results = _searcher.SearchArticles("", 20, yearMin: 2020,
            author: "Shared", journal: "Example Journal", corpusFilter: filter);
        Assert.Equal(AllowedKeys[1], Assert.Single(results).ArticleKey);
    }

    [Fact]
    public void LookupAndCountRespectTheSubset()
    {
        var filter = LexicalIndex.CreateKeyFilter([.. AllowedKeys, "https://openalex.org/W9999"]);
        Assert.Equal(82, _searcher.Count);
        Assert.Equal(2, _searcher.CountMatching(filter));
        Assert.NotNull(_searcher.GetByKey(AllowedKeys[0], filter));
        Assert.Null(_searcher.GetByKey("https://openalex.org/W1", filter));
        Assert.NotNull(_searcher.GetByKey("https://openalex.org/W1"));
    }

    [Fact]
    public void ConcurrentScopesRemainIndependent()
    {
        var first = LexicalIndex.CreateKeyFilter([AllowedKeys[0]]);
        var second = LexicalIndex.CreateKeyFilter([AllowedKeys[1]]);

        Parallel.For(0, 24, iteration =>
        {
            var filter = iteration % 2 == 0 ? first : second;
            var expected = AllowedKeys[iteration % 2];
            Assert.Equal(expected, Assert.Single(_searcher.Search("quartz", 20, corpusFilter: filter)).ArticleKey);
            Assert.Equal(20, _searcher.Search("quartz", 20).Count);
        });
    }

    [Fact]
    public void EmptyFilterCannotBecomeUnrestricted() =>
        Assert.Throws<ArgumentException>(() => LexicalIndex.CreateKeyFilter([]));

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void InvalidRerankConcurrencyFailsBeforeLoadingResources(int concurrency)
    {
        var options = new ServerOptions
        {
            IndexPath = "",
            CrossEncoderPath = "",
            MaxConcurrentReranks = concurrency,
        };
        Assert.Throws<ArgumentOutOfRangeException>(() => new RetrievalService(options));
    }

    [Fact]
    public void ConfigurationResolvesRelativeFilesAndNormalizesIds()
    {
        var configPath = WriteConfiguration(" W9001 \nhttps://openalex.org/W9001\nhttp://openalex.org/w9002\n\nW9999\n");
        var corpora = OpenAlexCorpus.Load(configPath);
        Assert.Same(OpenAlexCorpus.Full, corpora[0]);
        Assert.Equal("/mcp", corpora[0].McpPath);
        var subset = corpora[1];
        Assert.Equal("/mcp/mit", subset.McpPath);
        Assert.Equal(3, subset.RequestedIds);
        Assert.Equal(2, _searcher.CountMatching(subset.Filter!));
    }

    [Theory]
    [InlineData("")]
    [InlineData(" \n\n")]
    [InlineData("W9001\nnot-a-work")]
    [InlineData("https://example.com/W9001")]
    public void InvalidIdFilesFailClosed(string ids)
    {
        var configPath = WriteConfiguration(ids);
        Assert.Throws<InvalidDataException>(() => OpenAlexCorpus.Load(configPath));
    }

    [Theory]
    [InlineData("../mit")]
    [InlineData("mit/extra")]
    [InlineData("MIT")]
    [InlineData("openalex")]
    [InlineData("")]
    public void InvalidOrReservedNamesAreRejected(string name)
    {
        var configPath = WriteConfiguration("W9001", name);
        Assert.Throws<InvalidDataException>(() => OpenAlexCorpus.Load(configPath));
    }

    [Fact]
    public void MissingFilesFailClosed()
    {
        var configPath = WriteConfiguration("W9001");
        File.Delete(Path.Combine(_directory, "ids.txt"));
        Assert.Throws<FileNotFoundException>(() => OpenAlexCorpus.Load(configPath));
        Assert.Throws<FileNotFoundException>(() => OpenAlexCorpus.Load(Path.Combine(_directory, "missing.json")));
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"customCorporas\":[]}")]
    [InlineData("{\"customCorpora\":[{\"name\":\"mit\",\"idFile\":\"ids.txt\"}]}")]
    public void MisspelledConfigurationIsRejected(string json)
    {
        var configPath = Path.Combine(_directory, "corpora.json");
        File.WriteAllText(configPath, json);
        Assert.Throws<JsonException>(() => OpenAlexCorpus.Load(configPath));
    }

    [Fact]
    public void DuplicateNamesAreRejected()
    {
        var configPath = WriteConfiguration("W9001");
        File.WriteAllText(configPath, """
            {"customCorpora":[{"name":"mit","idsFile":"ids.txt"},{"name":"mit","idsFile":"ids.txt"}]}
            """);
        Assert.Throws<InvalidDataException>(() => OpenAlexCorpus.Load(configPath));
    }

    [Fact]
    public void EmptyConfigurationKeepsTheFullEndpoint()
    {
        var configPath = Path.Combine(_directory, "corpora.json");
        File.WriteAllText(configPath, "{\"customCorpora\":[]}");
        Assert.Same(OpenAlexCorpus.Full, Assert.Single(OpenAlexCorpus.Load(configPath)));
    }

    private string WriteConfiguration(string ids, string name = "mit")
    {
        File.WriteAllText(Path.Combine(_directory, "ids.txt"), ids);
        var configPath = Path.Combine(_directory, "corpora.json");
        File.WriteAllText(configPath, JsonSerializer.Serialize(new
        {
            customCorpora = new[] { new { name, idsFile = "ids.txt" } },
        }));
        return configPath;
    }

    private static void AddWork(IndexWriter writer, string key, string title, string body, int year, int citations)
    {
        var metadata = new BibliographicMetadata("", "", "Shared Author", "", "Example Journal", "", "en",
            "article", citations, "", "", "", "", "", "");
        writer.AddDocument(LexicalIndex.CreateDocument(new ArticleDocument(
            key, key, title, body, year, "", "", false, metadata)));
    }

    public void Dispose()
    {
        _searcher.Dispose();
        System.IO.Directory.Delete(_directory, recursive: true);
    }
}