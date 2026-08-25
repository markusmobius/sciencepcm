using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using SciencePcm.Embed;

namespace SciencePcm.Search;

public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        Options options;
        try
        {
            options = Options.Parse(args);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }

        var session = TextEmbedder.CreateSession(options.ModelDirectory, options.Threads, options.UseGpu);
        var embedder = new TextEmbedder(
            new EmbedderOptions(options.ModelDirectory, options.Threads, options.MaxTokens),
            session);

        using var searcher = new DenseSearcher(
            options.IndexPath,
            options.KeysPath,
            embedder,
            options.View,
            options.FetchMultiplier);

        Console.WriteLine($"index      : {options.IndexPath}");
        Console.WriteLine($"vectors    : {searcher.Size:N0} x {searcher.Dimensions}");
        Console.WriteLine($"model      : {options.ModelDirectory}");
        Console.WriteLine($"max tokens : {options.MaxTokens}");
        Console.WriteLine();

        return options.QueriesPath is null
            ? RunSingle(searcher, options)
            : RunBatch(searcher, options);
    }

    private static int RunSingle(DenseSearcher searcher, Options options)
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

    private static int RunBatch(DenseSearcher searcher, Options options)
    {
        var queries = LoadQueries(options.QueriesPath!, options.Limit);
        Console.WriteLine($"Searching {queries.Count:N0} queries at k={options.K} ...");

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(options.RunPath!))!);
        using var writer = new StreamWriter(options.RunPath!);

        var stopwatch = Stopwatch.StartNew();
        var latencies = new List<double>(queries.Count);

        for (var i = 0; i < queries.Count; i++)
        {
            var query = queries[i];
            var started = stopwatch.Elapsed.TotalMilliseconds;
            var hits = searcher.SearchArticles(query.QueryText, options.K);
            latencies.Add(stopwatch.Elapsed.TotalMilliseconds - started);

            writer.WriteLine(JsonSerializer.Serialize(new RunRecord
            {
                QueryId = query.QueryId,
                Hits = hits.Select(h => h.ArticleKey).ToList(),
                Scores = hits.Select(h => h.Score).ToList(),
            }));

            if ((i + 1) % 100 == 0)
            {
                Console.WriteLine($"  {i + 1:N0}/{queries.Count:N0}");
            }
        }

        stopwatch.Stop();
        latencies.Sort();

        Console.WriteLine();
        Console.WriteLine($"Queries    : {queries.Count:N0}");
        Console.WriteLine($"Elapsed    : {stopwatch.Elapsed}");
        Console.WriteLine($"Latency p50: {Percentile(latencies, 0.50):N1} ms");
        Console.WriteLine($"Latency p95: {Percentile(latencies, 0.95):N1} ms");
        Console.WriteLine();
        Console.WriteLine($"Run written to {options.RunPath}");
        Console.WriteLine("Score it with: python eval/run_eval.py --retriever runfile --run <file> --qrels <qrels>");
        return 0;
    }

    private static double Percentile(List<double> sorted, double fraction)
    {
        if (sorted.Count == 0) return 0;
        var position = (int)Math.Clamp(Math.Round(fraction * (sorted.Count - 1)), 0, sorted.Count - 1);
        return sorted[position];
    }

    private static List<QueryRecord> LoadQueries(string path, int? limit)
    {
        var queries = new List<QueryRecord>();

        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var record = JsonSerializer.Deserialize<QueryRecord>(line)
                ?? throw new InvalidOperationException($"Could not parse: {line}");
            queries.Add(record);

            if (limit is int max && queries.Count >= max) break;
        }

        return queries;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
        Dense retrieval over a usearch index.

          --index <file>          usearch index. Required.
          --keys <file>           Id list. Default: <index without extension>.keys.txt
          --model <dir>           Query encoder ONNX directory. Required.
          --query "<text>"        Search one query and print the ranking.
          --queries <jsonl>       Batch mode over qrels (query_id, query_text).
          --run <file>            Where batch mode writes its run file. Required with --queries.
          --k <n>                 Results per query. Default: 100
          --fetch <n>             Passage over-fetch before collapsing to articles. Default: 4
          --max-tokens <n>        Query truncation. Default: 64
          --threads <n>           ONNX intra-op threads. Default: 8
          --limit <n>             Only the first n queries.
          --load                  Load the index into RAM instead of memory-mapping it.
          --gpu                   Use CUDA. Needs -p:UseGpu=true at build time.

        The query encoder is NOT the article encoder. MedCPT ships separate weights and
        using the article encoder for queries silently degrades every result.
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

public sealed class Options
{
    public string IndexPath { get; private set; } = "";
    public string KeysPath { get; private set; } = "";
    public string ModelDirectory { get; private set; } = "";
    public string? Query { get; private set; }
    public string? QueriesPath { get; private set; }
    public string? RunPath { get; private set; }
    public int K { get; private set; } = 100;
    public int FetchMultiplier { get; private set; } = 4;
    public int MaxTokens { get; private set; } = 64;
    public int Threads { get; private set; } = 8;
    public int? Limit { get; private set; }
    public bool View { get; private set; } = true;
    public bool UseGpu { get; private set; }

    public static Options Parse(string[] args)
    {
        var options = new Options();
        var index = 0;

        string Next()
        {
            if (index + 1 >= args.Length) throw new ArgumentException($"{args[index]} needs a value.");
            return args[++index];
        }

        for (; index < args.Length; index++)
        {
            switch (args[index])
            {
                case "--index": options.IndexPath = Next(); break;
                case "--keys": options.KeysPath = Next(); break;
                case "--model": options.ModelDirectory = Next(); break;
                case "--query": options.Query = Next(); break;
                case "--queries": options.QueriesPath = Next(); break;
                case "--run": options.RunPath = Next(); break;
                case "--k": options.K = int.Parse(Next()); break;
                case "--fetch": options.FetchMultiplier = int.Parse(Next()); break;
                case "--max-tokens": options.MaxTokens = int.Parse(Next()); break;
                case "--threads": options.Threads = int.Parse(Next()); break;
                case "--limit": options.Limit = int.Parse(Next()); break;
                case "--load": options.View = false; break;
                case "--gpu": options.UseGpu = true; break;
                default: throw new ArgumentException($"Unknown argument '{args[index]}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(options.IndexPath)) throw new ArgumentException("--index is required.");
        if (string.IsNullOrWhiteSpace(options.ModelDirectory)) throw new ArgumentException("--model is required.");
        if (options.Query is null && options.QueriesPath is null) throw new ArgumentException("Pass --query or --queries.");
        if (options.QueriesPath is not null && options.RunPath is null) throw new ArgumentException("--queries needs --run.");

        if (string.IsNullOrWhiteSpace(options.KeysPath))
        {
            var directory = Path.GetDirectoryName(Path.GetFullPath(options.IndexPath))!;
            var stem = Path.GetFileNameWithoutExtension(options.IndexPath);
            options.KeysPath = Path.Combine(directory, stem + ".keys.txt");
        }

        if (!File.Exists(options.IndexPath)) throw new FileNotFoundException($"No index at {options.IndexPath}");
        if (!File.Exists(options.KeysPath)) throw new FileNotFoundException($"No key file at {options.KeysPath}");

        return options;
    }
}
