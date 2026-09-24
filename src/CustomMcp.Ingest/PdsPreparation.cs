using System.Security.Cryptography;
using System.Text.Json;
using Pds;
using SciencePcm.Core;
using SciencePcm.Index;

namespace CustomMcp.Ingest;

public static class PdsPreparation
{
    public static async Task<PreparedPdsCorpus> PrepareAsync(PdsCorpus corpus, string inputPath,
        string dataRoot, string indexRoot, int shardSize = 1000, ChunkOptions? chunkOptions = null, int threads = 8)
    {
        PdsCorpus.ValidateName(corpus.Name);
        PdsCorpus.ValidateCloudPath(corpus.InputCloudPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(threads, 1);
        var options = chunkOptions ?? ChunkOptions.Default;
        var inputHash = await PdsIngestor.HashAsync(inputPath);
        var revision = Fingerprint(new { PdsIngestor.SchemaVersion, corpus.Name, corpus.InputCloudPath, inputHash, shardSize, options });
        var dataDirectory = Path.Combine(Path.GetFullPath(dataRoot), corpus.Name, revision);
        var reportPath = Path.Combine(dataDirectory, "ingest-report.json");
        if (!Directory.Exists(dataDirectory))
        {
            using var reader = new PdsReader();
            reader.Open(inputPath);
            var result = await PdsIngestor.WriteAsync(reader, inputPath, corpus.InputCloudPath, dataDirectory, corpus.Name, shardSize, options);
            if (result.SourceSha256 != inputHash) throw new InvalidDataException("The PDS changed before ingestion began.");
        }

        using var report = JsonDocument.Parse(await File.ReadAllTextAsync(reportPath));
        var root = report.RootElement;
        if (root.GetProperty("schema_version").GetInt32() != PdsIngestor.SchemaVersion ||
            root.GetProperty("corpus_name").GetString() != corpus.Name ||
            root.GetProperty("input_sha256").GetString() != inputHash ||
            root.GetProperty("input_cloud_path").GetString() != corpus.InputCloudPath)
            throw new InvalidDataException($"Ingest report does not match '{corpus.Name}'.");
        var counts = root.GetProperty("counts");
        var shards = counts.GetProperty("shards_written").GetInt32();
        foreach (var (directory, prefix) in new[] { ("abstracts", "abstracts"), ("passages", "articles"), ("passages", "chunks") })
        {
            if (Directory.GetFiles(Path.Combine(dataDirectory, directory), prefix + "-part-*.parquet").Length != shards)
                throw new InvalidDataException($"Incomplete {prefix} shards for '{corpus.Name}'.");
        }

        var prepared = new PreparedPdsCorpus
        {
            Name = corpus.Name,
            InputCloudPath = corpus.InputCloudPath,
            InputSha256 = inputHash,
            Generation = Fingerprint(new { revision, LexicalIndex.SchemaVersion }),
            Documents = counts.GetProperty("articles_written").GetInt32(),
            Passages = counts.GetProperty("chunks_written").GetInt32(),
        };
        var indexDirectory = prepared.IndexDirectory(indexRoot);
        var metadata = Path.Combine(dataDirectory, "passages", "articles-part-*.parquet");
        foreach (var (schema, directory, prefix) in new[] { ("abstracts", "abstracts", "abstracts"), ("chunks", "passages", "chunks") })
        {
            var exitCode = await SciencePcm.Index.Program.Main(["build", "--schema", schema,
                "--input", Path.Combine(dataDirectory, directory, prefix + "-part-*.parquet"), "--metadata", metadata,
                "--out", Path.Combine(indexDirectory, directory), "--threads", threads.ToString(), "--ram-buffer", "256"]);
            if (exitCode != 0) throw new IOException($"Index build failed for '{corpus.Name}/{directory}'.");
        }
        ValidateIndexes(prepared, indexRoot);
        if (await PdsIngestor.HashAsync(inputPath) != inputHash) throw new InvalidDataException("The PDS changed during preparation.");
        await prepared.PublishAsync(indexRoot);
        Console.WriteLine($"{corpus.Name}: {prepared.Documents:N0} papers, {prepared.Passages:N0} passages; ready at {corpus.McpPath}");
        return prepared;
    }

    internal static void ValidateIndexes(PreparedPdsCorpus prepared, string indexRoot)
    {
        var indexDirectory = prepared.IndexDirectory(indexRoot);
        using var papers = new LexicalSearcher(Path.Combine(indexDirectory, "abstracts"));
        using var passages = new LexicalSearcher(Path.Combine(indexDirectory, "passages"));
        if (papers.Count != prepared.Documents || passages.Count != prepared.Passages)
            throw new InvalidDataException($"Index counts do not match the prepared corpus for '{prepared.Name}'.");
    }

    private static string Fingerprint<T>(T value) =>
        Convert.ToHexString(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(value))).ToLowerInvariant();
}