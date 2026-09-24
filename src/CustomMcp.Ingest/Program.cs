using SciencePcm.Core;

namespace CustomMcp.Ingest;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        try
        {
            if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
            {
                Console.WriteLine("""
                    CustomMcp PDS preparation.

                    prepare --data-root <directory> --index-root <directory> [options]
                    check --index-root <directory> [--config <file>]

                    --config <file>       Default: custom_mcps/pds.json beside the executable.
                    --collection <name>   Prepare/check one configured collection; default: all.
                    --cache <directory>   Default: <data-root>/cache.
                    --input <PDS file>    Local source override; requires exactly one selected collection.
                    --offline            Use existing cached PDS files without contacting the cloud.
                    --threads <n>        Indexing threads. Default: 8.
                    --shard-size <n>     Papers per shard. Default: 1000.
                    --target-words <n>   Passage target words. Default: 300.
                    --overlap-words <n>  Passage overlap. Default: 50.

                    Preparation refreshes each cloud file, reuses unchanged shards/indexes, and publishes
                    current.json only after both indexes pass their count checks. Restart the server to
                    load the new configuration or generation. MIT-specific repair is a separate tool.
                    """);
                return args.Length == 0 ? 1 : 0;
            }
            if (args[0] is not ("prepare" or "check")) throw new ArgumentException("Use prepare or check.");
            var configPath = Path.Combine(AppContext.BaseDirectory, "custom_mcps", "pds.json");
            string? dataRoot = null, indexRoot = null, cache = null, input = null, collection = null;
            var offline = false;
            var threads = 8;
            var shardSize = 1000;
            var target = 300;
            var overlap = 50;
            for (var index = 1; index < args.Length; index++)
            {
                var flag = args[index];
                string Next() => ++index < args.Length && !args[index].StartsWith("--", StringComparison.Ordinal)
                    ? args[index] : throw new ArgumentException($"Missing value for {flag}.");
                switch (flag)
                {
                    case "--config": configPath = Next(); break;
                    case "--data-root": dataRoot = Next(); break;
                    case "--index-root": indexRoot = Next(); break;
                    case "--cache": cache = Next(); break;
                    case "--input": input = Next(); break;
                    case "--collection": collection = Next(); break;
                    case "--offline": offline = true; break;
                    case "--threads": threads = int.Parse(Next()); break;
                    case "--shard-size": shardSize = int.Parse(Next()); break;
                    case "--target-words": target = int.Parse(Next()); break;
                    case "--overlap-words": overlap = int.Parse(Next()); break;
                    default: throw new ArgumentException($"Unknown option: {flag}");
                }
            }
            if (string.IsNullOrWhiteSpace(indexRoot)) throw new ArgumentException("--index-root is required.");
            if (threads < 1 || shardSize < 1 || target < ChunkOptions.Default.MinWords || overlap < 0 || overlap >= target)
                throw new ArgumentException("Invalid threads, shard size, or chunk sizes.");
            var corpora = PdsCorpus.Load(configPath);
            if (collection is not null)
            {
                corpora = corpora.Where(corpus => corpus.Name == collection).ToArray();
                if (corpora.Count == 0) throw new ArgumentException($"Unknown collection: {collection}");
            }
            if (args[0] == "check")
            {
                foreach (var corpus in corpora)
                {
                    var prepared = PreparedPdsCorpus.Load(corpus, indexRoot);
                    PdsPreparation.ValidateIndexes(prepared, indexRoot);
                    Console.WriteLine($"{corpus.Name} {corpus.McpPath}: {prepared.Documents:N0} papers, {prepared.Passages:N0} passages; {corpus.InputCloudPath}");
                }
                return 0;
            }
            if (string.IsNullOrWhiteSpace(dataRoot)) throw new ArgumentException("--data-root is required for prepare.");
            if (input is not null && (corpora.Count != 1 || offline))
                throw new ArgumentException("--input requires exactly one selected collection and cannot be combined with --offline.");
            cache ??= Path.Combine(dataRoot, "cache");
            foreach (var corpus in corpora)
            {
                var path = input ?? (offline ? CloudPdsSource.CachedPath(corpus.InputCloudPath, cache)
                    : await CloudPdsSource.DownloadAsync(corpus.InputCloudPath, cache, refresh: true));
                if (!File.Exists(path)) throw new FileNotFoundException($"PDS input is missing for '{corpus.Name}'.", path);
                await PdsPreparation.PrepareAsync(corpus, path, dataRoot, indexRoot, shardSize, new ChunkOptions(target, overlap), threads);
            }
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"CustomMcp preparation failed: {exception.Message}");
            return 1;
        }
    }
}