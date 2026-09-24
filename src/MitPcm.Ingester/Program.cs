using System.Text.Json;
using CustomMcp.Ingest;
using Pds;
using SciencePcm.Core;

namespace MitPcm.Ingester;

internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        try
        {
            if (args.Contains("--help") || args.Contains("-h"))
            {
                Console.WriteLine("""
                    MIT PDS ingestion using the shared SciencePCM Parquet schema.

                    --out <new directory>        Write passages/, abstracts/, and ingest-report.json.
                    --config <file>              Default: config.json beside the executable.
                    --cache <directory>          Default: runs/mit-ingest/cache.
                    --input <local PDS>          Use a local input instead of downloading the configured cloud file.
                    --shard-size <n>             Papers per shard. Default: 1000.
                    --target-words <n>           Target passage size. Default: 300.
                    --overlap-words <n>          Passage overlap. Default: 50.
                    --inspect                    Inspect and validate without writing Parquet.
                    --prepare-repair <directory> Prepare reference metadata (optional --limit <n>).
                    --write-repair <directory>   Write a reviewed corrected PDS.
                    --upload-repair <directory>  Upload and verify an approved corrected PDS.
                    """);
                return 0;
            }
            var configPath = Path.Combine(AppContext.BaseDirectory, "config.json");
            var cacheDirectory = Path.GetFullPath(Path.Combine("runs", "mit-ingest", "cache"));
            string? inputPath = null;
            string? repairDirectory = null;
            string? writeRepairDirectory = null;
            string? uploadRepairDirectory = null;
            string? outputDirectory = null;
            var inspect = false;
            var shardSize = 1000;
            var targetWords = 300;
            var overlapWords = 50;
            int? limit = null;
            for (var argumentIndex = 0; argumentIndex < args.Length; argumentIndex++)
            {
                var flag = args[argumentIndex];
                string Next() => ++argumentIndex < args.Length
                    ? args[argumentIndex]
                    : throw new ArgumentException($"Missing value for {flag}.");
                switch (flag)
                {
                    case "--config": configPath = Path.GetFullPath(Next()); break;
                    case "--cache": cacheDirectory = Path.GetFullPath(Next()); break;
                    case "--input": inputPath = Path.GetFullPath(Next()); break;
                    case "--prepare-repair": repairDirectory = Path.GetFullPath(Next()); break;
                    case "--write-repair": writeRepairDirectory = Path.GetFullPath(Next()); break;
                    case "--upload-repair": uploadRepairDirectory = Path.GetFullPath(Next()); break;
                    case "--out": outputDirectory = Path.GetFullPath(Next()); break;
                    case "--shard-size": shardSize = int.Parse(Next()); break;
                    case "--target-words": targetWords = int.Parse(Next()); break;
                    case "--overlap-words": overlapWords = int.Parse(Next()); break;
                    case "--limit": limit = int.Parse(Next()); break;
                    case "--inspect": inspect = true; break;
                    default: throw new ArgumentException($"Unknown argument: {flag}");
                }
            }
            var modeCount = new[] { repairDirectory, writeRepairDirectory, uploadRepairDirectory, outputDirectory }
                .Count(value => value is not null) + (inspect ? 1 : 0);
            if (modeCount != 1)
                throw new ArgumentException("Choose exactly one of --out, --inspect, --prepare-repair, --write-repair, or --upload-repair.");
            if (limit is not null && repairDirectory is null)
                throw new ArgumentException("--limit is only supported with --prepare-repair.");
            if (writeRepairDirectory is not null && (repairDirectory is not null || limit is not null))
                throw new ArgumentException("--write-repair cannot be combined with --prepare-repair or --limit.");
            if (uploadRepairDirectory is not null &&
                (repairDirectory is not null || writeRepairDirectory is not null || limit is not null || inputPath is not null))
                throw new ArgumentException("--upload-repair cannot be combined with other repair modes, --input, or --limit.");
            if (uploadRepairDirectory is not null)
                return await PdsRepair.UploadAsync(uploadRepairDirectory);

            using var config = JsonDocument.Parse(await File.ReadAllTextAsync(configPath));
            var cloudPath = config.RootElement.GetProperty("inputCloudPath").GetString()
                ?? throw new InvalidDataException("inputCloudPath is required.");
            inputPath ??= await DownloadAsync(cloudPath, cacheDirectory);

            Console.WriteLine($"Input: {inputPath}");
            Console.WriteLine($"Bytes: {new FileInfo(inputPath).Length:N0}");
            using var reader = new PdsReader();
            reader.Open(inputPath);
            if (repairDirectory is not null)
                return await PdsRepair.PrepareAsync(reader, inputPath, repairDirectory, limit);
            if (writeRepairDirectory is not null)
                return await PdsRepair.WriteAsync(reader, inputPath, writeRepairDirectory);
            if (outputDirectory is not null)
            {
                var result = await PdsIngestor.WriteAsync(reader, inputPath, cloudPath, outputDirectory,
                    "mit", shardSize, new ChunkOptions(targetWords, overlapWords));
                Console.WriteLine($"Ingested: {result.Articles:N0} papers; {result.Abstracts:N0} abstracts; {result.Chunks:N0} chunks; {result.Shards:N0} shards.");
                Console.WriteLine($"Output: {outputDirectory}");
                return 0;
            }

            var keyPaths = reader.GetKeyPaths().ToArray();
            Console.WriteLine($"Compression: {reader.CompressionMode}; records: {keyPaths.Length:N0}");
            Console.WriteLine($"Metadata fields: {string.Join(", ", reader.Metadata.EnumerateObject().Select(property => property.Name))}");
            foreach (var name in new[] { "journal", "article_count", "failed_paper_count", "requested_article_count", "companion" })
            {
                if (!reader.Metadata.TryGetProperty(name, out var value)) continue;
                Console.WriteLine($"Store {name}:");
                Inspect(value, "  ", 0);
            }
            if (reader.Metadata.TryGetProperty("checkpoint_records", out var checkpoints))
            {
                Console.WriteLine($"Checkpoint shape: {checkpoints.ValueKind}");
                if (checkpoints.ValueKind == JsonValueKind.Array)
                    foreach (var checkpoint in checkpoints.EnumerateArray().Take(1)) Inspect(checkpoint, "  ", 0);
                else if (checkpoints.ValueKind == JsonValueKind.Object)
                    foreach (var checkpoint in checkpoints.EnumerateObject().Take(1)) Inspect(checkpoint.Value, "  ", 0);
            }
            foreach (var keyPath in keyPaths.Take(2))
            {
                Console.WriteLine($"Key: {string.Join(" / ", keyPath)}");
                Inspect(reader.ReadKey<JsonPdsValue>(keyPath).Value, "  ", 0);
            }
            var parsedPapers = new List<ParsedArticle>();
            var totalChunks = 0;
            foreach (var keyPath in keyPaths)
            {
                var key = string.Join('/', keyPath);
                var paper = reader.ReadKey<JsonPdsValue>(keyPath).Value;
                var parsed = PdsPaperParser.Parse(paper, key, cloudPath + "#" + key, "mit");
                parsedPapers.Add(parsed);
                totalChunks += Chunker.Chunk(parsed).Count;
            }
            Console.WriteLine($"Papers with labelled abstract blocks: {parsedPapers.Count(paper => paper.Blocks.Any(block => block.Kind == SectionKind.Abstract))}");
            foreach (var parsed in parsedPapers.Take(3))
            {
                Console.WriteLine($"Leading sections: {parsed.Article.Title}");
                Console.WriteLine("Headings: " + string.Join(" | ", parsed.Blocks.GroupBy(block => block.SectionIndex)
                    .Take(8).Select(group => group.First().SectionTitle ?? "(untitled)")));
                foreach (var section in parsed.Blocks.GroupBy(block => block.SectionIndex).Take(2))
                {
                    var text = string.Join(" ", section.Select(block => block.Text));
                    Console.WriteLine($"[{section.First().SectionTitle}] {text[..Math.Min(text.Length, 1800)]}");
                }
            }
            Console.WriteLine($"Mapped: {parsedPapers.Count:N0} papers; {totalChunks:N0} chunks; " +
                $"abstracts: {parsedPapers.Count(paper => paper.Article.Abstract is not null):N0}; " +
                $"dated: {parsedPapers.Count(paper => paper.Article.PubYear is not null):N0}; " +
                $"journal: {parsedPapers.Count(paper => paper.Article.Journal is not null):N0}.");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"MIT ingest failed: {exception.Message}");
            return 1;
        }
    }

    private static void Inspect(JsonElement value, string prefix, int depth)
    {
        if (depth > 5) return;
        if (value.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in value.EnumerateObject().Take(30))
            {
                Console.WriteLine($"{prefix}{property.Name}: {property.Value.ValueKind}");
                Inspect(property.Value, prefix + "  ", depth + 1);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
        {
            Console.WriteLine($"{prefix}Count: {value.GetArrayLength()}");
            foreach (var item in value.EnumerateArray().Take(1)) Inspect(item, prefix + "  ", depth + 1);
        }
        else
        {
            var text = value.ToString().Replace('\n', ' ').Replace('\r', ' ');
            Console.WriteLine(prefix + text[..Math.Min(text.Length, 100)]);
        }
    }

    internal static Task<string> DownloadAsync(string cloudPath, string cacheDirectory) =>
        CloudPdsSource.DownloadAsync(cloudPath, cacheDirectory);
}

internal sealed class JsonPdsValue : IPdsJsonValue
{
    public JsonElement Value { get; private set; }

    public void ReadJson(JsonElement element) => Value = element.Clone();

    public void WriteJson(Utf8JsonWriter writer) => Value.WriteTo(writer);
}