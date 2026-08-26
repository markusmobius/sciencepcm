using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lucene.Net.Index;
using Lucene.Net.Search.Similarities;
using Lucene.Net.Store;
using SciencePcm.Embed;

namespace SciencePcm.Lexical;

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        try
        {
            return args[0] switch
            {
                "build" => await BuildAsync(BuildOptions.Parse(args)),
                "search" => SearchCommand(SearchOptions.Parse(args)),
                _ => throw new ArgumentException($"Unknown command '{args[0]}'. Use build or search."),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }

    private static async Task<int> BuildAsync(BuildOptions options)
    {
        Console.WriteLine($"input   : {options.Input}");
        Console.WriteLine($"schema  : {options.Schema}");
        Console.WriteLine($"out     : {options.Output}");
        Console.WriteLine();

        System.IO.Directory.CreateDirectory(options.Output);

        using var analyzer = LexicalIndex.CreateAnalyzer();
        using var directory = FSDirectory.Open(options.Output);

        var config = new IndexWriterConfig(LexicalIndex.Version, analyzer)
        {
            OpenMode = OpenMode.CREATE,
            RAMBufferSizeMB = options.RamBufferMb,
            Similarity = new BM25Similarity(),
        };

        using var writer = new IndexWriter(directory, config);

        var queue = new BlockingCollection<ArticleRecord>(options.Threads * 4096);
        var stopwatch = Stopwatch.StartNew();
        var indexed = 0L;
        var skipped = 0L;

        var workers = Enumerable.Range(0, options.Threads).Select(_ => Task.Run(() =>
        {
            // IndexWriter is thread-safe, so writers share one instance rather than
            // building separate indexes that would then need merging.
            foreach (var record in queue.GetConsumingEnumerable())
            {
                writer.AddDocument(LexicalIndex.CreateDocument(new ArticleDocument(
                    record.Id,
                    record.ArticleKey,
                    record.Title,
                    record.Body,
                    record.Year,
                    record.Pmid,
                    record.Section,
                    record.IsRetracted,
                    record.Metadata)));

                var count = Interlocked.Increment(ref indexed);
                if (count % 500_000 == 0)
                {
                    var rate = count / stopwatch.Elapsed.TotalSeconds;
                    Console.WriteLine($"  {count:N0} indexed  ({rate:N0}/s)");
                }
            }
        })).ToArray();

        await foreach (var record in ParquetTextSource.ReadArticlesAsync(options.Input, options.Schema))
        {
            if (string.IsNullOrWhiteSpace(record.Body))
            {
                skipped++;
                continue;
            }

            queue.Add(record);

            if (options.Limit is long max && indexed + skipped >= max) break;
        }

        queue.CompleteAdding();
        await Task.WhenAll(workers);

        Console.WriteLine("\nCommitting ...");
        writer.Commit();

        if (options.Optimize)
        {
            Console.WriteLine("Merging to a single segment (slow, improves query speed) ...");
            writer.ForceMerge(1);
            writer.Commit();
        }

        stopwatch.Stop();

        var bytes = new DirectoryInfo(options.Output).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
        Console.WriteLine();
        Console.WriteLine($"Documents indexed : {indexed:N0}");
        Console.WriteLine($"Skipped (empty)   : {skipped:N0}");
        Console.WriteLine($"Index size        : {bytes / 1024.0 / 1024 / 1024:N2} GB");
        Console.WriteLine($"Elapsed           : {stopwatch.Elapsed}");
        return 0;
    }

    private static int SearchCommand(SearchOptions options)
    {
        using var searcher = new LexicalSearcher(options.IndexPath, options.FetchMultiplier);
        Console.WriteLine($"index     : {options.IndexPath}");
        Console.WriteLine($"documents : {searcher.Count:N0}");
        Console.WriteLine();

        if (options.QueriesPath is null)
        {
            var stopwatch = Stopwatch.StartNew();
            var hits = searcher.SearchArticles(options.Query!, options.K);
            stopwatch.Stop();

            Console.WriteLine($"\"{options.Query}\"  ({stopwatch.Elapsed.TotalMilliseconds:N1} ms)");
            Console.WriteLine();

            var rank = 1;
            foreach (var hit in hits)
            {
                Console.WriteLine($"{rank,4}. {hit.Score:F4}  {hit.ArticleKey}");
                rank++;
            }
            return 0;
        }

        var queries = LoadQueries(options.QueriesPath, options.Limit);
        Console.WriteLine($"Searching {queries.Count:N0} queries at k={options.K} ...");

        var results = new List<string>[queries.Count];
        var scores = new List<float>[queries.Count];
        var latencies = new double[queries.Count];
        var progress = 0;

        Parallel.For(0, queries.Count, new ParallelOptions { MaxDegreeOfParallelism = options.Threads }, i =>
        {
            var started = Stopwatch.GetTimestamp();
            var hits = searcher.SearchArticles(queries[i].QueryText, options.K);
            latencies[i] = Stopwatch.GetElapsedTime(started).TotalMilliseconds;

            results[i] = hits.Select(h => h.ArticleKey).ToList();
            scores[i] = hits.Select(h => h.Score).ToList();

            var done = Interlocked.Increment(ref progress);
            if (done % 100 == 0) Console.WriteLine($"  {done:N0}/{queries.Count:N0}");
        });

        System.IO.Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(options.RunPath!))!);
        using (var stream = new StreamWriter(options.RunPath!))
        {
            for (var i = 0; i < queries.Count; i++)
            {
                stream.WriteLine(JsonSerializer.Serialize(new RunRecord
                {
                    QueryId = queries[i].QueryId,
                    Hits = results[i],
                    Scores = scores[i],
                }));
            }
        }

        Array.Sort(latencies);
        Console.WriteLine();
        Console.WriteLine($"Queries    : {queries.Count:N0}");
        Console.WriteLine($"Latency p50: {latencies[latencies.Length / 2]:N1} ms");
        Console.WriteLine($"Latency p95: {latencies[(int)(latencies.Length * 0.95)]:N1} ms");
        Console.WriteLine($"\nRun written to {options.RunPath}");
        return 0;
    }

    private static List<QueryRecord> LoadQueries(string path, int? limit)
    {
        var queries = new List<QueryRecord>();

        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            queries.Add(JsonSerializer.Deserialize<QueryRecord>(line)
                ?? throw new InvalidOperationException($"Could not parse: {line}"));
            if (limit is int max && queries.Count >= max) break;
        }

        return queries;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
        BM25 retrieval over the ingest Parquet, using Lucene.NET.

        build --input <glob> --out <dir> [options]
          --schema abstracts|openalex|chunks   Default: abstracts
          --threads <n>               Indexing threads. Default: 8
          --ram-buffer <mb>           Writer buffer. Default: 512
          --optimize                  Merge to one segment. Slow to build, faster to query.
          --limit <n>                 Stop after n documents.

        search --index <dir> [options]
          --query "<text>"            Search one query and print the ranking.
          --queries <jsonl>           Batch over qrels (query_id, query_text).
          --run <file>                Where batch mode writes its run file.
          --k <n>                     Results per query. Default: 100
          --fetch <n>                 Over-fetch before collapsing to articles. Default: 4
          --threads <n>               Parallel queries. Default: 8
          --limit <n>                 Only the first n queries.
        """);
    }
}

public sealed class QueryRecord
{
    [JsonPropertyName("query_id")] public string QueryId { get; set; } = "";
    [JsonPropertyName("query_text")] public string QueryText { get; set; } = "";
}

public sealed class RunRecord
{
    [JsonPropertyName("query_id")] public string QueryId { get; set; } = "";
    [JsonPropertyName("hits")] public List<string> Hits { get; set; } = [];
    [JsonPropertyName("scores")] public List<float> Scores { get; set; } = [];
}

public sealed class BuildOptions
{
    public string Input { get; private set; } = "";
    public string Output { get; private set; } = "";
    public CorpusSchema Schema { get; private set; } = CorpusSchema.Abstracts;
    public int Threads { get; private set; } = 8;
    public double RamBufferMb { get; private set; } = 512;
    public bool Optimize { get; private set; }
    public long? Limit { get; private set; }

    public static BuildOptions Parse(string[] args)
    {
        var options = new BuildOptions();

        for (var i = 1; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");

            switch (args[i])
            {
                case "--input": options.Input = Next(); break;
                case "--out": options.Output = Next(); break;
                case "--schema":
                    options.Schema = Next().ToLowerInvariant() switch
                    {
                        "abstracts" => CorpusSchema.Abstracts,
                        "openalex" => CorpusSchema.OpenAlex,
                        "chunks" => CorpusSchema.Chunks,
                        var other => throw new ArgumentException($"Unknown schema '{other}'."),
                    };
                    break;
                case "--threads": options.Threads = int.Parse(Next()); break;
                case "--ram-buffer": options.RamBufferMb = double.Parse(Next()); break;
                case "--optimize": options.Optimize = true; break;
                case "--limit": options.Limit = long.Parse(Next()); break;
                default: throw new ArgumentException($"Unknown argument '{args[i]}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(options.Input)) throw new ArgumentException("--input is required.");
        if (string.IsNullOrWhiteSpace(options.Output)) throw new ArgumentException("--out is required.");
        return options;
    }
}

public sealed class SearchOptions
{
    public string IndexPath { get; private set; } = "";
    public string? Query { get; private set; }
    public string? QueriesPath { get; private set; }
    public string? RunPath { get; private set; }
    public int K { get; private set; } = 100;
    public int FetchMultiplier { get; private set; } = 4;
    public int Threads { get; private set; } = 8;
    public int? Limit { get; private set; }

    public static SearchOptions Parse(string[] args)
    {
        var options = new SearchOptions();

        for (var i = 1; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");

            switch (args[i])
            {
                case "--index": options.IndexPath = Next(); break;
                case "--query": options.Query = Next(); break;
                case "--queries": options.QueriesPath = Next(); break;
                case "--run": options.RunPath = Next(); break;
                case "--k": options.K = int.Parse(Next()); break;
                case "--fetch": options.FetchMultiplier = int.Parse(Next()); break;
                case "--threads": options.Threads = int.Parse(Next()); break;
                case "--limit": options.Limit = int.Parse(Next()); break;
                default: throw new ArgumentException($"Unknown argument '{args[i]}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(options.IndexPath)) throw new ArgumentException("--index is required.");
        if (options.Query is null && options.QueriesPath is null) throw new ArgumentException("Pass --query or --queries.");
        if (options.QueriesPath is not null && options.RunPath is null) throw new ArgumentException("--queries needs --run.");
        return options;
    }
}
