using System.Security.Cryptography;
using System.Text.Json;
using CustomMcp.Ingest;
using Parquet.Serialization;
using Pds;
using SciencePcm.Core;
using SciencePcm.Embed;
using SciencePcm.Index;
using SciencePcm.Ingest;
using SciencePcm.Server;
using Xunit;

namespace MitPcm.Tests;

[Collection("MIT corpus")]
public sealed class IngestionTests(MitCorpusFixture fixture)
{
    [Fact]
    public void ClosingOneCollectionDoesNotDisposeSharedReranker()
    {
        var options = new ServerOptions
        {
            IndexPath = fixture.PaperIndex, PassageIndexPath = fixture.PassageIndex,
            CrossEncoderPath = fixture.Model, Threads = 1, RerankBatch = 2, MaxConcurrentReranks = 1,
        };
        using var reranker = new SharedReranker(options);
        using var second = new RetrievalService(options, reranker);
        using (var first = new RetrievalService(options, reranker))
            Assert.Equal(MitCorpusFixture.PaperKey, Assert.Single(first.Search("neurons", 1)).ArticleKey);
        Assert.Equal(MitCorpusFixture.PaperKey, Assert.Single(second.Search("neurons", 1)).ArticleKey);
    }

    [Fact]
    public async Task PreparationIsIdempotentAndRejectsChangedConfiguration()
    {
        var root = Path.Combine(fixture.Root, "prepared");
        var corpus = new PdsCorpus { Name = "prepared", InputCloudPath = "test/source.pds" };
        var prepared = await PdsPreparation.PrepareAsync(corpus, fixture.Input, root + "/data", root + "/index", threads: 1);
        var pointer = Path.Combine(root, "index", corpus.Name, "current.json");
        var modified = File.GetLastWriteTimeUtc(pointer);
        var repeated = await PdsPreparation.PrepareAsync(corpus, fixture.Input, root + "/data", root + "/index", threads: 1);
        Assert.Equal(prepared, repeated);
        Assert.Equal(modified, File.GetLastWriteTimeUtc(pointer));
        Assert.Equal(prepared, PreparedPdsCorpus.Load(corpus, root + "/index"));
        Assert.Throws<InvalidDataException>(() => PreparedPdsCorpus.Load(corpus with { InputCloudPath = "test/replaced.pds" }, root + "/index"));
        using var papers = new LexicalSearcher(Path.Combine(prepared.IndexDirectory(root + "/index"), "abstracts"));
        Assert.NotNull(papers.GetByKey("prepared:paper-one"));
    }

    [Fact]
    public async Task FailedRefreshKeepsLastPreparedGeneration()
    {
        var root = Path.Combine(fixture.Root, "failed-refresh");
        var corpus = new PdsCorpus { Name = "refresh", InputCloudPath = "test/source.pds" };
        var prepared = await PdsPreparation.PrepareAsync(corpus, fixture.Input, root + "/data", root + "/index", threads: 1);
        var input = Path.Combine(root, "invalid.pds");
        MitCorpusFixture.CreateInput(input, invalidTitle: true);
        await Assert.ThrowsAsync<InvalidDataException>(() => PdsPreparation.PrepareAsync(corpus, input, root + "/data", root + "/index", threads: 1));
        Assert.Equal(prepared, PreparedPdsCorpus.Load(corpus, root + "/index"));
    }

    [Theory]
    [InlineData("abstracts")]
    [InlineData("passages")]
    public async Task PreflightRejectsMismatchedCountsAndMissingIndexes(string directory)
    {
        var root = Path.Combine(fixture.Root, "preflight-" + directory);
        var indexRoot = Path.Combine(root, "index");
        var corpus = new PdsCorpus { Name = "preflight", InputCloudPath = "test/source.pds" };
        var prepared = await PdsPreparation.PrepareAsync(corpus, fixture.Input, root + "/data", indexRoot, threads: 1);
        var configPath = Path.Combine(root, "config.json");
        await File.WriteAllTextAsync(configPath, JsonSerializer.Serialize(new
        {
            customCorpora = new[] { new { name = corpus.Name, inputCloudPath = corpus.InputCloudPath } },
        }));
        string[] arguments = ["check", "--config", configPath, "--index-root", indexRoot];
        Assert.Equal(0, await CustomMcp.Ingest.Program.Main(arguments));

        var mismatched = directory == "abstracts"
            ? prepared with { Documents = prepared.Documents + 1 }
            : prepared with { Passages = prepared.Passages + 1 };
        await mismatched.PublishAsync(indexRoot);
        Assert.Equal(1, await CustomMcp.Ingest.Program.Main(arguments));

        await prepared.PublishAsync(indexRoot);
        Directory.Delete(Path.Combine(prepared.IndexDirectory(indexRoot), directory), recursive: true);
        Assert.Equal(1, await CustomMcp.Ingest.Program.Main(arguments));
    }

    [Fact]
    public void SameSourceKeyIsNamespacedByCollection()
    {
        using var json = JsonDocument.Parse("""{"metadata":{"title":"A paper","doi":"10.1234/example"},"abstract":[],"body":{"sections":[]}}""");
        var first = PdsPaperParser.Parse(json.RootElement, "same-key", "source.pds", "first");
        var second = PdsPaperParser.Parse(json.RootElement, "same-key", "source.pds", "second");
        Assert.Equal("first:same-key", first.Article.ArticleKey);
        Assert.Equal("second:same-key", second.Article.ArticleKey);
        Assert.Equal("first", first.Article.SourceCorpus);
        Assert.Equal("10.1234/example", first.Article.Doi);
    }

    [Theory]
    [InlineData("papers", "collection/input.pds", true)]
    [InlineData("neural-computation_2", "sciencepcm/Sept222026/input_2026_09_23_fixed.pds", true)]
    [InlineData("../papers", "collection/input.pds", false)]
    [InlineData("Papers", "collection/input.pds", false)]
    [InlineData("papers", "/collection/input.pds", false)]
    [InlineData("papers", "collection/../input.pds", false)]
    [InlineData("papers", "collection/input.pds|", false)]
    [InlineData("papers", "collection/input.json", false)]
    public async Task NamedPdsConfigurationValidatesBeforeDataAccess(string name, string cloudPath, bool valid)
    {
        var path = Path.Combine(fixture.Root, Guid.NewGuid().ToString("N") + ".json");
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(new
        {
            customCorpora = new[] { new { name, inputCloudPath = cloudPath } },
        }));
        if (valid)
        {
            var corpus = Assert.Single(PdsCorpus.Load(path));
            Assert.Equal(name, corpus.Name);
            Assert.Equal(cloudPath, corpus.InputCloudPath);
            Assert.Equal("/mcp/" + name, corpus.McpPath);
        }
        else Assert.Throws<InvalidDataException>(() => PdsCorpus.Load(path));
    }

    [Theory]
    [InlineData("{\"customCorpora\":[]}")]
    [InlineData("{\"customCorpora\":null}")]
    [InlineData("{\"customCorpora\":[null]}")]
    [InlineData("{\"customCorpora\":[{\"name\":\"papers\",\"inputCloudPath\":\"a/input.pds\"},{\"name\":\"papers\",\"inputCloudPath\":\"b/input.pds\"}]}")]
    public async Task NamedPdsConfigurationRejectsEmptyOrDuplicateCollections(string json)
    {
        var path = Path.Combine(fixture.Root, Guid.NewGuid().ToString("N") + ".json");
        await File.WriteAllTextAsync(path, json);
        Assert.Throws<InvalidDataException>(() => PdsCorpus.Load(path));
    }

    [Fact]
    public async Task ParquetRoundTripPreservesAuthorsAbstractDoiAndMissingDate()
    {
        var articles = new List<ArticleRow>();
        foreach (var path in ParquetTextSource.ExpandGlob(fixture.MetadataGlob))
        {
            await using var stream = File.OpenRead(path);
            articles.AddRange((await ParquetSerializer.DeserializeAsync<ArticleRow>(stream)).Data);
        }
        Assert.Equal(2, articles.Count);
        var paper = Assert.Single(articles, article => article.ArticleKey == MitCorpusFixture.PaperKey);
        Assert.Equal(MitCorpusFixture.Abstract, paper.Abstract);
        Assert.Equal("Luis Sa-Couto; Andreas Wichert", paper.Authors);
        Assert.Equal("10.1162/neco_a_01243", paper.Doi);
        Assert.Equal("Neural Computation", paper.Journal);
        Assert.Equal(2024, paper.PubYear);
        Assert.Equal("test/source.pds#paper-one", paper.SourcePath);
        var erratum = Assert.Single(articles, article => article.ArticleKey == MitCorpusFixture.ErratumKey);
        Assert.Null(erratum.Abstract);
        Assert.Null(erratum.PubYear);
        Assert.Equal("10.1162/neco_c_01397", erratum.Doi);
    }

    [Fact]
    public async Task MetadataJoinPreservesTitleOnlyPapersWithoutChangingDefaultReader()
    {
        var joined = await ReadArticles(fixture.AbstractGlob, CorpusSchema.Abstracts, fixture.MetadataGlob);
        Assert.Equal(2, joined.Count);
        Assert.Equal("", Assert.Single(joined, article => article.Id == MitCorpusFixture.ErratumKey).Body);
        var paper = Assert.Single(joined, article => article.Id == MitCorpusFixture.PaperKey);
        Assert.Equal("Luis Sa-Couto; Andreas Wichert", paper.Metadata!.Authors);
        Assert.Equal("https://doi.org/10.1162/neco_a_01243", paper.Metadata.LandingPageUrl);
        Assert.Null(paper.Metadata.CitedByCount);
        Assert.Null(paper.Metadata.IsOpenAccess);
        var unjoined = await ReadArticles(fixture.AbstractGlob, CorpusSchema.Abstracts);
        Assert.Equal(MitCorpusFixture.PaperKey, Assert.Single(unjoined).Id);
    }

    [Fact]
    public async Task SharedWriterKeepsItsExistingAbstractFilteringDefault()
    {
        var output = Path.Combine(fixture.Root, "default-writer");
        Directory.CreateDirectory(output);
        var count = await CorpusShardWriter.WriteAsync(output, output, 0,
            [new ArticleRow { ArticleKey = "full", Title = "Full paper", Abstract = "A real abstract." },
             new ArticleRow { ArticleKey = "title", Title = "Title only" }], []);
        Assert.Equal(1, count);
        Assert.Single(await ReadArticles(Path.Combine(output, "abstracts-part-*.parquet"), CorpusSchema.Abstracts));
    }

    [Fact]
    public async Task IndexesContainAllPapersAndOnlyEvidencePassages()
    {
        using var papers = new LexicalSearcher(fixture.PaperIndex);
        using var passages = new LexicalSearcher(fixture.PassageIndex);
        Assert.Equal(2, papers.Count);
        Assert.NotNull(papers.GetByKey(MitCorpusFixture.ErratumKey));
        Assert.Equal(MitCorpusFixture.PaperKey, Assert.Single(papers.SearchArticles("", 5, author: "Wichert")).ArticleKey);
        Assert.Equal(fixture.Result.Chunks, passages.Count);
        var records = await ReadArticles(fixture.ChunkGlob, CorpusSchema.Chunks, fixture.MetadataGlob);
        Assert.DoesNotContain(records, record => record.Body.Contains("must not become", StringComparison.Ordinal));
        Assert.Contains(records, record => record.Section == "Methods" && record.Metadata!.Authors.Contains("Wichert", StringComparison.Ordinal));
        Assert.Equal(records.Count, records.Select(record => record.Id).Distinct().Count());
    }

    [Fact]
    public async Task ReportAndSourceHashDescribeTheCompletedCorpus()
    {
        using var report = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(fixture.Output, "ingest-report.json")));
        var counts = report.RootElement.GetProperty("counts");
        Assert.Equal(2, counts.GetProperty("articles_written").GetInt32());
        Assert.Equal(1, counts.GetProperty("abstracts_written").GetInt32());
        Assert.Equal(2, counts.GetProperty("shards_written").GetInt32());
        Assert.Equal(fixture.Result.Chunks, counts.GetProperty("chunks_written").GetInt32());
        Assert.Equal(0, counts.GetProperty("failed").GetInt32());
        var hash = Convert.ToHexString(SHA256.HashData(await File.ReadAllBytesAsync(fixture.Input))).ToLowerInvariant();
        Assert.Equal(hash, report.RootElement.GetProperty("input_sha256").GetString());
    }

    [Fact]
    public async Task ExistingOutputIsNeverOverwritten()
    {
        var reportPath = Path.Combine(fixture.Output, "ingest-report.json");
        var before = await File.ReadAllBytesAsync(reportPath);
        using var reader = new PdsReader();
        reader.Open(fixture.Input);
        await Assert.ThrowsAsync<IOException>(() => PdsIngestor.WriteAsync(reader, fixture.Input, "test/source.pds", fixture.Output, "papers"));
        Assert.Equal(before, await File.ReadAllBytesAsync(reportPath));
    }

    [Fact]
    public async Task InvalidRecordCannotPublishPartialShards()
    {
        var input = Path.Combine(fixture.Root, "invalid.pds");
        var output = Path.Combine(fixture.Root, "invalid-output");
        MitCorpusFixture.CreateInput(input, invalidTitle: true);
        using var reader = new PdsReader();
        reader.Open(input);
        await Assert.ThrowsAsync<InvalidDataException>(() => PdsIngestor.WriteAsync(reader, input, "test/invalid.pds", output, "papers", shardSize: 1));
        Assert.False(Directory.Exists(output));
        Assert.Empty(Directory.EnumerateDirectories(fixture.Root, "invalid-output.*.partial"));
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(100, -1)]
    [InlineData(100, 100)]
    [InlineData(10, 0)]
    public async Task InvalidChunkOptionsFailBeforeWriting(int target, int overlap)
    {
        using var reader = new PdsReader();
        reader.Open(fixture.Input);
        var output = Path.Combine(fixture.Root, Guid.NewGuid().ToString("N"));
        await Assert.ThrowsAsync<ArgumentException>(() => PdsIngestor.WriteAsync(reader, fixture.Input, "test/source.pds", output, "papers",
            chunkOptions: new ChunkOptions(target, overlap)));
        Assert.False(Directory.Exists(output));
    }

    private static async Task<List<ArticleRecord>> ReadArticles(string glob, CorpusSchema schema, string? metadata = null)
    {
        var result = new List<ArticleRecord>();
        await foreach (var record in ParquetTextSource.ReadArticlesAsync(glob, schema, metadata)) result.Add(record);
        return result;
    }
}