using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using Parquet;
using Parquet.Serialization;
using Parquet.Serialization.Attributes;

namespace OpenAlex.Ingest;

internal static class Program
{
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNameCaseInsensitive = false,
    };

    private static async Task<int> Main(string[] args)
    {
        Options options;
        try
        {
            options = Options.Parse(args);
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            Console.Error.WriteLine(Options.Usage);
            return 1;
        }

        var files = Directory.EnumerateFiles(options.Input, "*.gz", SearchOption.AllDirectories)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToList();
        if (files.Count == 0)
        {
            Console.Error.WriteLine($"No .gz files found under {options.Input}.");
            return 1;
        }

        Directory.CreateDirectory(options.Output);
        Console.WriteLine($"OpenAlex works files : {files.Count:N0}");
        Console.WriteLine($"Output               : {options.Output}");

        var stopwatch = Stopwatch.StartNew();
        var stats = new IngestStats();
        var channel = Channel.CreateBounded<OpenAlexAbstractRow>(new BoundedChannelOptions(options.Threads * 1024)
        {
            SingleReader = true,
            FullMode = BoundedChannelFullMode.Wait,
        });
        var writer = WriteAsync(channel.Reader, options, stats);

        await Parallel.ForEachAsync(
            files,
            new ParallelOptions { MaxDegreeOfParallelism = options.Threads },
            async (path, cancellationToken) =>
            {
                try
                {
                    await ReadFileAsync(path, channel.Writer, options, stats, cancellationToken);
                }
                catch (Exception ex)
                {
                    Interlocked.Increment(ref stats.FilesFailed);
                    Console.Error.WriteLine($"FAIL {path}: {ex.GetType().Name}: {ex.Message}");
                }

                var complete = Interlocked.Increment(ref stats.FilesRead);
                if (complete % 100 == 0 || complete == files.Count)
                {
                    var rate = Volatile.Read(ref stats.RecordsRead) / Math.Max(1, stopwatch.Elapsed.TotalSeconds);
                    Console.WriteLine($"  {complete:N0}/{files.Count:N0} files, {stats.RecordsRead:N0} works ({rate:N0}/s)");
                }
            });

        channel.Writer.Complete();
        await writer;
        stopwatch.Stop();

        await WriteReportAsync(options, files.Count, stats, stopwatch.Elapsed);
        Console.WriteLine();
        Console.WriteLine($"Works read          : {stats.RecordsRead:N0}");
        Console.WriteLine($"Abstracts written   : {stats.AbstractsWritten:N0}");
        Console.WriteLine($"Without abstract    : {stats.WithoutAbstract:N0}");
        Console.WriteLine($"Invalid records     : {stats.InvalidRecords:N0}");
        Console.WriteLine($"Files failed        : {stats.FilesFailed:N0}");
        Console.WriteLine($"Elapsed             : {stopwatch.Elapsed}");

        return stats.FilesFailed == 0 ? 0 : 2;
    }

    private static async Task ReadFileAsync(
        string path,
        ChannelWriter<OpenAlexAbstractRow> writer,
        Options options,
        IngestStats stats,
        CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(path);
        await using var gzip = new GZipStream(file, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip);

        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            var ordinal = Interlocked.Increment(ref stats.RecordsClaimed);
            if (options.Limit is long limit && ordinal > limit) break;
            Interlocked.Increment(ref stats.RecordsRead);

            try
            {
                var work = JsonSerializer.Deserialize<OpenAlexWork>(line, Json);
                if (work is null || string.IsNullOrWhiteSpace(work.Id))
                {
                    Interlocked.Increment(ref stats.InvalidRecords);
                    continue;
                }

                var abstractText = ReconstructAbstract(work.AbstractInvertedIndex);
                if (string.IsNullOrWhiteSpace(abstractText))
                {
                    Interlocked.Increment(ref stats.WithoutAbstract);
                    continue;
                }

                await writer.WriteAsync(new OpenAlexAbstractRow
                {
                    openalex_id = work.Id,
                    title = work.Title ?? work.DisplayName,
                    @abstract = abstractText,
                    publication_year = work.PublicationYear,
                    pmid = work.Ids?.Pmid,
                    doi = work.Doi,
                    language = work.Language,
                    type = work.Type,
                    is_retracted = work.IsRetracted,
                }, cancellationToken);
            }
            catch (JsonException)
            {
                Interlocked.Increment(ref stats.InvalidRecords);
            }
        }
    }

    internal static string? ReconstructAbstract(Dictionary<string, int[]>? invertedIndex)
    {
        if (invertedIndex is null || invertedIndex.Count == 0) return null;

        var lastPosition = invertedIndex.Values
            .Where(positions => positions.Length > 0)
            .SelectMany(positions => positions)
            .DefaultIfEmpty(-1)
            .Max();
        if (lastPosition < 0) return null;

        var words = new string?[lastPosition + 1];
        foreach (var (word, positions) in invertedIndex)
        {
            foreach (var position in positions)
            {
                if ((uint)position < (uint)words.Length) words[position] = word;
            }
        }

        return string.Join(' ', words.Where(word => word is not null));
    }

    private static async Task WriteAsync(
        ChannelReader<OpenAlexAbstractRow> reader,
        Options options,
        IngestStats stats)
    {
        var rows = new List<OpenAlexAbstractRow>(options.ShardSize);
        var shard = 0;

        await foreach (var row in reader.ReadAllAsync())
        {
            rows.Add(row);
            if (rows.Count >= options.ShardSize)
            {
                await FlushAsync(options.Output, shard++, rows, stats);
            }
        }

        if (rows.Count > 0) await FlushAsync(options.Output, shard, rows, stats);
    }

    private static async Task FlushAsync(
        string output,
        int shard,
        List<OpenAlexAbstractRow> rows,
        IngestStats stats)
    {
        var path = Path.Combine(output, $"abstracts-part-{shard:D5}.parquet");
        await using var stream = File.Create(path);
        await ParquetSerializer.SerializeAsync(rows, stream, new ParquetOptions
        {
            CompressionMethod = CompressionMethod.Zstd,
            CompressionLevel = System.IO.Compression.CompressionLevel.Optimal,
        });

        stats.AbstractsWritten += rows.Count;
        Console.WriteLine($"  wrote {Path.GetFileName(path)}: {rows.Count:N0} abstracts");
        rows.Clear();
    }

    private static async Task WriteReportAsync(Options options, int files, IngestStats stats, TimeSpan elapsed)
    {
        var report = new
        {
            dataset_name = "OpenAlex abstracts",
            created_at = DateTimeOffset.UtcNow,
            input = Path.GetFullPath(options.Input),
            output_schema = "openalex-abstracts-v1",
            shard_size = options.ShardSize,
            counts = new
            {
                files_discovered = files,
                files_failed = stats.FilesFailed,
                works_read = stats.RecordsRead,
                abstracts_written = stats.AbstractsWritten,
                without_abstract = stats.WithoutAbstract,
                invalid_records = stats.InvalidRecords,
            },
            elapsed_seconds = elapsed.TotalSeconds,
            peak_working_set_bytes = Process.GetCurrentProcess().PeakWorkingSet64,
        };

        await File.WriteAllTextAsync(
            Path.Combine(options.Output, "openalex-ingest-report.json"),
            JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
    }
}

internal sealed class OpenAlexWork
{
    [JsonPropertyName("id")] public string? Id { get; set; }
    [JsonPropertyName("doi")] public string? Doi { get; set; }
    [JsonPropertyName("title")] public string? Title { get; set; }
    [JsonPropertyName("display_name")] public string? DisplayName { get; set; }
    [JsonPropertyName("publication_year")] public int? PublicationYear { get; set; }
    [JsonPropertyName("ids")] public OpenAlexIds? Ids { get; set; }
    [JsonPropertyName("abstract_inverted_index")] public Dictionary<string, int[]>? AbstractInvertedIndex { get; set; }
    [JsonPropertyName("language")] public string? Language { get; set; }
    [JsonPropertyName("type")] public string? Type { get; set; }
    [JsonPropertyName("is_retracted")] public bool IsRetracted { get; set; }
}

internal sealed class OpenAlexIds
{
    [JsonPropertyName("pmid")] public string? Pmid { get; set; }
}

#pragma warning disable IDE1006
internal sealed class OpenAlexAbstractRow
{
    [ParquetRequired]
    public string openalex_id { get; set; } = "";
    public string? title { get; set; }
    public string? @abstract { get; set; }
    public int? publication_year { get; set; }
    public string? pmid { get; set; }
    public string? doi { get; set; }
    public string? language { get; set; }
    public string? type { get; set; }
    public bool is_retracted { get; set; }
}
#pragma warning restore IDE1006

internal sealed class IngestStats
{
    public long RecordsClaimed;
    public long RecordsRead;
    public long AbstractsWritten;
    public long WithoutAbstract;
    public long InvalidRecords;
    public int FilesRead;
    public int FilesFailed;
}

internal sealed class Options
{
    public const string Usage = """
        Usage:
          OpenAlex.Ingest --input <snapshot-data-dir> --out <dir> [options]

        Reads the OpenAlex works snapshot (*.gz JSONL), reconstructs abstracts from
        abstract_inverted_index, and writes compact Parquet shards.

        Options:
          --threads <n>       Files read concurrently. Default: processor count.
          --shard-size <n>    Abstracts per Parquet file. Default: 250000.
          --limit <n>         Read at most n works (smoke tests).
        """;

    public string Input { get; private set; } = "";
    public string Output { get; private set; } = "";
    public int Threads { get; private set; } = Environment.ProcessorCount;
    public int ShardSize { get; private set; } = 250_000;
    public long? Limit { get; private set; }

    public static Options Parse(string[] args)
    {
        var options = new Options();
        for (var i = 0; i < args.Length; i++)
        {
            var flag = args[i];
            string Next() => i + 1 < args.Length
                ? args[++i]
                : throw new ArgumentException($"Missing value for {flag}.");

            switch (flag)
            {
                case "--input": options.Input = Next(); break;
                case "--out": options.Output = Next(); break;
                case "--threads": options.Threads = int.Parse(Next()); break;
                case "--shard-size": options.ShardSize = int.Parse(Next()); break;
                case "--limit": options.Limit = long.Parse(Next()); break;
                case "--help":
                case "-h": throw new ArgumentException(Usage);
                default: throw new ArgumentException($"Unknown argument '{flag}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(options.Input)) throw new ArgumentException("--input is required.");
        if (!Directory.Exists(options.Input)) throw new ArgumentException($"Input directory not found: {options.Input}");
        if (string.IsNullOrWhiteSpace(options.Output)) throw new ArgumentException("--out is required.");
        if (options.Threads < 1) throw new ArgumentException("--threads must be positive.");
        if (options.ShardSize < 1) throw new ArgumentException("--shard-size must be positive.");
        return options;
    }
}