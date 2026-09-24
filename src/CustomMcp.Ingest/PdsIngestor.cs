using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Pds;
using SciencePcm.Core;
using SciencePcm.Ingest;

namespace CustomMcp.Ingest;

public static class PdsIngestor
{
    public const int SchemaVersion = 2;

    public static async Task<PdsIngestResult> WriteAsync(PdsReader reader, string inputPath, string sourcePath,
        string outputDirectory, string corpusName, int shardSize = 1000, ChunkOptions? chunkOptions = null)
    {
        PdsCorpus.ValidateName(corpusName);
        var options = chunkOptions ?? ChunkOptions.Default;
        if (shardSize <= 0) throw new ArgumentException("Shard size must be positive.");
        if (options.TargetWords <= 0 || options.MinWords <= 0 || options.MinWords > options.TargetWords ||
            options.OverlapWords < 0 || options.OverlapWords >= options.TargetWords)
            throw new ArgumentException("Chunk sizes require 0 <= overlap < target and 0 < minimum <= target.");
        outputDirectory = Path.GetFullPath(outputDirectory);
        if (Directory.Exists(outputDirectory) || File.Exists(outputDirectory))
            throw new IOException($"Output already exists; choose a new --out directory: {outputDirectory}");

        var paths = reader.GetKeyPaths().OrderBy(path => string.Join('/', path), StringComparer.Ordinal).ToArray();
        if (paths.Length == 0) throw new InvalidDataException("The PDS contains no papers.");
        var checkpoints = reader.Metadata.TryGetProperty("checkpoint_records", out var checkpointRecords)
            ? checkpointRecords.EnumerateArray().ToDictionary(record => record.GetProperty("paper_key").GetString()!, StringComparer.Ordinal)
            : new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        var journals = reader.Metadata.TryGetProperty("journal", out var journal) && journal.ValueKind == JsonValueKind.Array
            ? journal.EnumerateArray().Where(value => value.ValueKind == JsonValueKind.String).Select(value => value.GetString()!)
                .Where(value => !string.IsNullOrWhiteSpace(value) && !value.Equals("unknown", StringComparison.OrdinalIgnoreCase))
                .Distinct(StringComparer.Ordinal).ToArray()
            : [];
        var stopwatch = Stopwatch.StartNew();
        var sourceHash = await HashAsync(inputPath);
        var staging = outputDirectory + "." + Guid.NewGuid().ToString("N") + ".partial";
        var passagesDirectory = Path.Combine(staging, "passages");
        var abstractsDirectory = Path.Combine(staging, "abstracts");
        var articles = new List<ArticleRow>(shardSize);
        var chunks = new List<ChunkRow>();
        var articleCount = 0;
        var abstractCount = 0;
        var chunkCount = 0;
        var shardCount = 0;
        var missingDates = 0;
        var withoutAbstract = new List<string>();

        try
        {
            Directory.CreateDirectory(passagesDirectory);
            Directory.CreateDirectory(abstractsDirectory);
            foreach (var path in paths)
            {
                var key = string.Join('/', path);
                var parsed = PdsPaperParser.Parse(reader.ReadKey<PdsJsonValue>(path).Value, key, sourcePath + "#" + key, corpusName);
                var article = parsed.Article;
                if (article.Journal is null && journals.Length == 1) article.Journal = journals[0];
                if (article.Doi is null && checkpoints.TryGetValue(key, out var checkpoint) &&
                    checkpoint.TryGetProperty("filename", out var filename) && filename.ValueKind == JsonValueKind.String &&
                    Regex.IsMatch(filename.GetString()!, @"^neco_[ace]_[0-9]{5}\.pdf$", RegexOptions.CultureInvariant))
                    article.Doi = "10.1162/" + Path.GetFileNameWithoutExtension(filename.GetString());

                var passages = Chunker.Chunk(parsed, options);
                articles.Add(article);
                chunks.AddRange(passages);
                articleCount++;
                chunkCount += passages.Count;
                if (article.Abstract is not null) abstractCount++;
                else withoutAbstract.Add(article.ArticleKey);
                if (article.PubYear is null) missingDates++;
                if (articles.Count >= shardSize) await FlushAsync();
            }
            if (articles.Count > 0) await FlushAsync();
            if (await HashAsync(inputPath) != sourceHash)
                throw new InvalidDataException("The input PDS changed during ingestion.");
            var report = new
            {
                schema_version = SchemaVersion,
                corpus_name = corpusName,
                input_cloud_path = sourcePath,
                input_sha256 = sourceHash,
                chunking = new { target_words = options.TargetWords, overlap_words = options.OverlapWords, min_words = options.MinWords },
                counts = new
                {
                    articles_written = articleCount,
                    document_rows_written = articleCount,
                    abstracts_written = abstractCount,
                    chunks_written = chunkCount,
                    shards_written = shardCount,
                    missing_publication_dates = missingDates,
                    failed = 0,
                },
                articles_without_abstract = withoutAbstract,
                elapsed_seconds = stopwatch.Elapsed.TotalSeconds,
            };
            await File.WriteAllTextAsync(Path.Combine(staging, "ingest-report.json"),
                JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
            Directory.Move(staging, outputDirectory);
            return new PdsIngestResult(articleCount, abstractCount, chunkCount, shardCount, sourceHash);
        }
        finally
        {
            if (Directory.Exists(staging)) Directory.Delete(staging, recursive: true);
        }

        async Task FlushAsync()
        {
            await CorpusShardWriter.WriteAsync(passagesDirectory, abstractsDirectory, shardCount,
                articles, chunks, includeWithoutAbstract: true);
            Console.WriteLine($"  wrote shard {shardCount:D4}: {articles.Count:N0} papers, {chunks.Count:N0} chunks");
            shardCount++;
            articles.Clear();
            chunks.Clear();
        }
    }

    public static async Task<string> HashAsync(string path)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexString(await SHA256.HashDataAsync(stream)).ToLowerInvariant();
    }
}

public sealed record PdsIngestResult(int Articles, int Abstracts, int Chunks, int Shards, string SourceSha256);

internal sealed class PdsJsonValue : IPdsJsonValue
{
    public JsonElement Value { get; private set; }
    public void ReadJson(JsonElement element) => Value = element.Clone();
    public void WriteJson(Utf8JsonWriter writer) => Value.WriteTo(writer);
}