using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Cloud.Unum.USearch;

namespace SciencePcm.Index;

internal static class Program
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        Options options;
        try
        {
            options = Options.Parse(args);
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            Console.Error.WriteLine();
            Console.Error.WriteLine(Options.Usage);
            return 1;
        }

        return Build(options);
    }

    private static int Build(Options options)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(options.OutputIndex))!);

        var pairs = VectorShards.Pairs(options.VectorDirectory);
        var dimensions = options.Dimensions > 0
            ? options.Dimensions
            : VectorShards.InferDimensions(pairs[0].Vectors, pairs[0].Ids);

        Console.WriteLine($"shards      : {pairs.Count}");
        Console.WriteLine($"dimensions  : {dimensions}");
        Console.WriteLine($"metric      : {options.Metric} (vectors are L2-normalised, so cosine == dot)");
        Console.WriteLine($"scalar      : {options.Scalar}");
        Console.WriteLine($"connectivity: {options.Connectivity}  expansionAdd: {options.ExpansionAdd}");

        using var index = new USearchIndex(
            options.Metric,
            options.Scalar,
            (ulong)dimensions,
            (ulong)options.Connectivity,
            (ulong)options.ExpansionAdd,
            (ulong)options.ExpansionSearch,
            false);

        Console.WriteLine($"hardware    : {index.HardwareAcceleration()}");
        Console.WriteLine();

        // The corpora overlap, so the same chunk id can appear twice with identical text.
        // First occurrence wins; a duplicate key in the index would make id -> passage
        // lookups ambiguous.
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var keys = new List<ulong>(options.BatchSize);
        var vectors = new List<float[]>(options.BatchSize);
        var idWriter = new StreamWriter(options.OutputKeys, false, Encoding.UTF8);

        var stopwatch = Stopwatch.StartNew();
        ulong nextKey = 0;
        var duplicates = 0L;
        var read = 0L;

        try
        {
            foreach (var record in VectorShards.Read(options.VectorDirectory, dimensions))
            {
                read++;
                if (!seen.Add(record.Id))
                {
                    duplicates++;
                    continue;
                }

                keys.Add(nextKey++);
                vectors.Add(record.Vector);
                idWriter.WriteLine(record.Id);

                if (keys.Count >= options.BatchSize)
                {
                    index.Add(keys.ToArray(), vectors.ToArray());
                    keys.Clear();
                    vectors.Clear();

                    if (nextKey % 500_000 < (ulong)options.BatchSize)
                    {
                        var rate = nextKey / Math.Max(0.001, stopwatch.Elapsed.TotalSeconds);
                        Console.WriteLine($"  {nextKey:N0} indexed  ({rate:N0}/s, {duplicates:N0} duplicates skipped)");
                    }
                }

                if (options.Limit is { } limit && (long)nextKey >= limit) break;
            }

            if (keys.Count > 0)
            {
                index.Add(keys.ToArray(), vectors.ToArray());
            }
        }
        finally
        {
            idWriter.Dispose();
        }

        Console.WriteLine();
        Console.WriteLine($"Saving to {options.OutputIndex} ...");
        index.Save(options.OutputIndex);
        stopwatch.Stop();

        var report = new
        {
            created_at = DateTimeOffset.UtcNow,
            vector_directory = options.VectorDirectory,
            index_path = options.OutputIndex,
            keys_path = options.OutputKeys,
            dimensions,
            metric = options.Metric.ToString(),
            scalar = options.Scalar.ToString(),
            connectivity = options.Connectivity,
            expansion_add = options.ExpansionAdd,
            expansion_search = options.ExpansionSearch,
            vectors_read = read,
            duplicates_skipped = duplicates,
            vectors_indexed = (long)index.Size(),
            index_bytes = new FileInfo(options.OutputIndex).Length,
            elapsed_seconds = stopwatch.Elapsed.TotalSeconds,
        };

        File.WriteAllText(
            Path.ChangeExtension(options.OutputIndex, ".report.json"),
            JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine();
        Console.WriteLine($"Vectors read     : {read:N0}");
        Console.WriteLine($"Duplicates       : {duplicates:N0}");
        Console.WriteLine($"Vectors indexed  : {index.Size():N0}");
        Console.WriteLine($"Index size       : {new FileInfo(options.OutputIndex).Length / 1024.0 / 1024 / 1024:N2} GB");
        Console.WriteLine($"Elapsed          : {stopwatch.Elapsed}");
        Console.WriteLine();
        Console.WriteLine($"Key N of the index maps to line N+1 of {options.OutputKeys}.");
        return 0;
    }
}

internal sealed class Options
{
    public const string Usage = """
        Usage:
          SciencePcm.Index --vectors <dir> --out <index-path> [options]

        Required:
          --vectors <dir>        Directory of vectors-part-*.f32 / ids-part-*.txt.
          --out <path>           Destination .usearch index. A sibling .keys.txt and
                                 .report.json are written alongside.

        Index shape:
          --dimensions <n>       Default: inferred from the first shard.
          --metric cos|ip|l2     Default: cos
          --scalar f32|f16|i8    Storage precision. f16 halves memory for a small recall
                                 cost. Default: f32
          --connectivity <n>     HNSW graph degree. Default: 32
          --expansion-add <n>    Build-time candidate list. Default: 128
          --expansion-search <n> Query-time candidate list. Default: 64

        Other:
          --batch <n>            Vectors per Add call. Default: 8192
          --limit <n>            Index at most n vectors (smoke tests).
        """;

    public string VectorDirectory { get; private set; } = "";
    public string OutputIndex { get; private set; } = "";
    public string OutputKeys => Path.ChangeExtension(OutputIndex, ".keys.txt");
    public int Dimensions { get; private set; }
    public MetricKind Metric { get; private set; } = MetricKind.Cos;
    public ScalarKind Scalar { get; private set; } = ScalarKind.Float32;
    public int Connectivity { get; private set; } = 32;
    public int ExpansionAdd { get; private set; } = 128;
    public int ExpansionSearch { get; private set; } = 64;
    public int BatchSize { get; private set; } = 8192;
    public long? Limit { get; private set; }

    public static Options Parse(string[] args)
    {
        var options = new Options();

        for (var i = 0; i < args.Length; i++)
        {
            var flag = args[i];
            string Next()
            {
                if (i + 1 >= args.Length) throw new ArgumentException($"Missing value for {flag}.");
                return args[++i];
            }

            switch (flag)
            {
                case "--vectors": options.VectorDirectory = Next(); break;
                case "--out": options.OutputIndex = Next(); break;
                case "--dimensions": options.Dimensions = int.Parse(Next()); break;
                case "--metric":
                    options.Metric = Next().ToLowerInvariant() switch
                    {
                        "cos" => MetricKind.Cos,
                        "ip" => MetricKind.Ip,
                        "l2" => MetricKind.L2sq,
                        var other => throw new ArgumentException($"Unknown metric '{other}'."),
                    };
                    break;
                case "--scalar":
                    options.Scalar = Next().ToLowerInvariant() switch
                    {
                        "f32" => ScalarKind.Float32,
                        "f16" => ScalarKind.Float16,
                        "i8" => ScalarKind.Int8,
                        var other => throw new ArgumentException($"Unknown scalar '{other}'."),
                    };
                    break;
                case "--connectivity": options.Connectivity = int.Parse(Next()); break;
                case "--expansion-add": options.ExpansionAdd = int.Parse(Next()); break;
                case "--expansion-search": options.ExpansionSearch = int.Parse(Next()); break;
                case "--batch": options.BatchSize = int.Parse(Next()); break;
                case "--limit": options.Limit = long.Parse(Next()); break;
                default: throw new ArgumentException($"Unrecognised argument '{flag}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(options.VectorDirectory)) throw new ArgumentException("--vectors is required.");
        if (string.IsNullOrWhiteSpace(options.OutputIndex)) throw new ArgumentException("--out is required.");
        return options;
    }
}
