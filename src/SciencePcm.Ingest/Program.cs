using System.Diagnostics;
using System.Text.Json;
using System.Threading.Channels;
using Parquet;
using Parquet.Serialization;
using SciencePcm.Core;

namespace SciencePcm.Ingest;

internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;

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

        var files = EnumerateInputs(options).ToList();
        Console.WriteLine($"Discovered {files.Count:N0} XML files across {options.Inputs.Count} corpus root(s).");
        if (options.Limit is { } limit && files.Count > limit)
        {
            files = files.Take(limit).ToList();
            Console.WriteLine($"Limited to {files.Count:N0} files.");
        }

        if (options.Sample > 0)
        {
            Sample(files, options);
            return 0;
        }

        Directory.CreateDirectory(options.OutputDirectory);

        var stopwatch = Stopwatch.StartNew();
        var stats = new IngestStats();

        var channel = Channel.CreateBounded<ParsedResult>(new BoundedChannelOptions(options.Threads * 4)
        {
            SingleReader = true,
            FullMode = BoundedChannelFullMode.Wait,
        });

        var writer = Task.Run(() => WriteAsync(channel.Reader, options, stats));

        var chunkOptions = new ChunkOptions(options.TargetWords, options.OverlapWords);
        var processed = 0;

        // The corpora overlap: a preprint can appear in both biorxiv and biorxiv-supp, or
        // as both a preprint and a PMC article, and ArticleKey is the same in each case.
        // Without this the same passages are indexed twice under identical chunk ids.
        var seenArticles = new System.Collections.Concurrent.ConcurrentDictionary<string, byte>(
            StringComparer.Ordinal);

        await Parallel.ForEachAsync(
            files,
            new ParallelOptions { MaxDegreeOfParallelism = options.Threads },
            async (file, token) =>
            {
                try
                {
                    var parsed = JatsParser.Parse(file.Path, file.Corpus);
                    if (parsed is null)
                    {
                        Interlocked.Increment(ref stats.SkippedNotArticle);
                        return;
                    }

                    if (!InRange(parsed.Article.PubYear, options))
                    {
                        Interlocked.Increment(ref stats.SkippedOutOfRange);
                        return;
                    }

                    if (!parsed.Article.HasBody)
                    {
                        Interlocked.Increment(ref stats.SkippedNoBody);
                        return;
                    }

                    if (!seenArticles.TryAdd(parsed.Article.ArticleKey, 0))
                    {
                        Interlocked.Increment(ref stats.SkippedDuplicate);
                        return;
                    }

                    var chunks = Chunker.Chunk(parsed, chunkOptions);
                    await channel.Writer.WriteAsync(new ParsedResult(parsed.Article, chunks), token);
                }
                catch (Exception ex)
                {
                    var failures = Interlocked.Increment(ref stats.Failed);
                    if (failures <= 20)
                    {
                        Console.Error.WriteLine($"FAIL {file.Path}: {ex.GetType().Name}: {ex.Message}");
                    }
                }
                finally
                {
                    var done = Interlocked.Increment(ref processed);
                    if (done % 10_000 == 0)
                    {
                        var rate = done / Math.Max(1, stopwatch.Elapsed.TotalSeconds);
                        Console.WriteLine($"  {done:N0}/{files.Count:N0} files  ({rate:N0}/s)");
                    }
                }
            });

        channel.Writer.Complete();
        await writer;

        stopwatch.Stop();
        await WriteReportAsync(options, files.Count, stats, stopwatch.Elapsed);

        Console.WriteLine();
        Console.WriteLine($"Articles written : {stats.Articles:N0}");
        Console.WriteLine($"Chunks written   : {stats.Chunks:N0}");
        Console.WriteLine($"Skipped (range)  : {stats.SkippedOutOfRange:N0}");
        Console.WriteLine($"Skipped (no body): {stats.SkippedNoBody:N0}");
        Console.WriteLine($"Skipped (dup key): {stats.SkippedDuplicate:N0}");
        Console.WriteLine($"Failed           : {stats.Failed:N0}");
        Console.WriteLine($"Elapsed          : {stopwatch.Elapsed}");
        Console.WriteLine();
        Console.WriteLine("Section distribution:");
        foreach (var (kind, count) in stats.SectionKinds.OrderByDescending(kv => kv.Value))
        {
            Console.WriteLine($"  {kind,-18} {count,10:N0} chunks  {100.0 * count / Math.Max(1, stats.Chunks),5:F1}%");
        }

        return stats.Failed > files.Count / 100 ? 2 : 0;
    }

    /// <summary>Prints parsed passages for eyeballing extraction quality; writes nothing.</summary>
    private static void Sample(List<InputFile> files, Options options)
    {
        var chunkOptions = new ChunkOptions(options.TargetWords, options.OverlapWords);

        foreach (var file in files.Take(options.Sample))
        {
            var parsed = JatsParser.Parse(file.Path, file.Corpus);
            if (parsed is null) continue;

            var a = parsed.Article;
            Console.WriteLine(new string('=', 100));
            Console.WriteLine($"{a.ArticleKey}  {a.PubYear}  {a.Journal}");
            Console.WriteLine($"  title      : {a.Title}");
            Console.WriteLine($"  doi/pmid   : {a.Doi} / {a.Pmid}");
            Console.WriteLine($"  type       : {a.ArticleType}   retracted={a.IsRetracted}");
            Console.WriteLine($"  license    : {a.LicenseUrl}");
            Console.WriteLine($"  body words : {a.BodyWordCount:N0}   sections={a.SectionCount}   refs={a.ReferenceCount}");

            var chunks = Chunker.Chunk(parsed, chunkOptions);
            Console.WriteLine($"  chunks     : {chunks.Count}");

            foreach (var group in chunks.GroupBy(c => c.SectionKind))
            {
                Console.WriteLine($"     {group.Key,-16} {group.Count(),4} chunks, {group.Sum(c => c.WordCount),7:N0} words");
            }

            foreach (var chunk in chunks.Take(3))
            {
                Console.WriteLine();
                Console.WriteLine($"  [{chunk.SectionKind}] {chunk.SectionPath} ({chunk.WordCount}w)");
                Console.WriteLine($"  {chunk.Text}");
            }
            Console.WriteLine();
        }
    }

    private static bool InRange(int? year, Options options)
    {
        if (options.YearMin is null && options.YearMax is null) return true;
        if (year is null) return false;
        if (options.YearMin is { } min && year < min) return false;
        if (options.YearMax is { } max && year > max) return false;
        return true;
    }

    private static IEnumerable<InputFile> EnumerateInputs(Options options)
    {
        foreach (var (corpus, root) in options.Inputs)
        {
            if (!Directory.Exists(root))
            {
                throw new DirectoryNotFoundException($"Input root not found: {root}");
            }

            foreach (var path in Directory.EnumerateFiles(root, "*.xml", SearchOption.AllDirectories))
            {
                yield return new InputFile(corpus, path);
            }
        }
    }

    private static async Task WriteAsync(ChannelReader<ParsedResult> reader, Options options, IngestStats stats)
    {
        var articles = new List<ArticleRow>(options.ShardSize);
        var chunks = new List<ChunkRow>(options.ShardSize * 16);
        var shard = 0;

        await foreach (var result in reader.ReadAllAsync())
        {
            articles.Add(result.Article);
            chunks.AddRange(result.Chunks);

            foreach (var chunk in result.Chunks)
            {
                stats.SectionKinds.TryGetValue(chunk.SectionKind, out var seen);
                stats.SectionKinds[chunk.SectionKind] = seen + 1;
                stats.SectionWords.TryGetValue(chunk.SectionKind, out var words);
                stats.SectionWords[chunk.SectionKind] = words + chunk.WordCount;
            }

            if (articles.Count >= options.ShardSize)
            {
                await FlushAsync(options, shard++, articles, chunks, stats);
            }
        }

        if (articles.Count > 0)
        {
            await FlushAsync(options, shard, articles, chunks, stats);
        }
    }

    private static async Task FlushAsync(
        Options options,
        int shard,
        List<ArticleRow> articles,
        List<ChunkRow> chunks,
        IngestStats stats)
    {
        var serializerOptions = new ParquetOptions
        {
            CompressionMethod = CompressionMethod.Zstd,
            CompressionLevel = System.IO.Compression.CompressionLevel.Optimal,
        };

        var articlePath = Path.Combine(options.OutputDirectory, $"articles-part-{shard:D4}.parquet");
        var chunkPath = Path.Combine(options.OutputDirectory, $"chunks-part-{shard:D4}.parquet");

        await using (var stream = File.Create(articlePath))
        {
            await ParquetSerializer.SerializeAsync(articles, stream, serializerOptions);
        }

        await using (var stream = File.Create(chunkPath))
        {
            await ParquetSerializer.SerializeAsync(chunks, stream, serializerOptions);
        }

        stats.Articles += articles.Count;
        stats.Chunks += chunks.Count;
        Console.WriteLine($"  wrote shard {shard:D4}: {articles.Count:N0} articles, {chunks.Count:N0} chunks");

        articles.Clear();
        chunks.Clear();
    }

    private static async Task WriteReportAsync(Options options, int discovered, IngestStats stats, TimeSpan elapsed)
    {
        var report = new
        {
            dataset_name = "SciencePCM JATS passage index source",
            created_at = DateTimeOffset.UtcNow,
            inputs = options.Inputs.Select(i => new { corpus = i.Corpus, root = i.Root }),
            year_min = options.YearMin,
            year_max = options.YearMax,
            chunking = new
            {
                target_words = options.TargetWords,
                overlap_words = options.OverlapWords,
                min_words = ChunkOptions.Default.MinWords,
            },
            counts = new
            {
                files_discovered = discovered,
                articles_written = stats.Articles,
                chunks_written = stats.Chunks,
                skipped_out_of_range = stats.SkippedOutOfRange,
                skipped_no_body = stats.SkippedNoBody,
                skipped_not_article = stats.SkippedNotArticle,
                skipped_duplicate_article_key = stats.SkippedDuplicate,
                failed = stats.Failed,
            },
            elapsed_seconds = elapsed.TotalSeconds,
            peak_working_set_bytes = Process.GetCurrentProcess().PeakWorkingSet64,
            section_kinds = stats.SectionKinds
                .OrderByDescending(kv => kv.Value)
                .ToDictionary(
                    kv => kv.Key,
                    kv => new
                    {
                        chunks = kv.Value,
                        words = stats.SectionWords.GetValueOrDefault(kv.Key),
                        chunk_percent = Math.Round(100.0 * kv.Value / Math.Max(1, stats.Chunks), 2),
                    }),
        };

        var path = Path.Combine(options.OutputDirectory, "ingest-report.json");
        await File.WriteAllTextAsync(
            path,
            JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"Report: {path}");
    }
}

internal sealed record InputFile(string Corpus, string Path);

internal sealed record ParsedResult(ArticleRow Article, List<ChunkRow> Chunks);

internal sealed class IngestStats
{
    public long Articles;
    public long Chunks;
    public int SkippedOutOfRange;
    public int SkippedNoBody;
    public int SkippedNotArticle;
    public int SkippedDuplicate;
    public int Failed;

    // Written only by the single-reader writer task.
    public Dictionary<string, long> SectionKinds { get; } = [];
    public Dictionary<string, long> SectionWords { get; } = [];
}

internal sealed record CorpusInput(string Corpus, string Root);

internal sealed class Options
{
    public const string Usage = """
        Usage:
          SciencePcm.Ingest --input <corpus>=<dir> [--input ...] --out <dir> [options]

        Options:
          --input <corpus>=<dir>   Corpus label and full-text root. Repeatable.
          --out <dir>              Output directory for Parquet shards.
          --year-min <n>           Drop articles published before this year.
          --year-max <n>           Drop articles published after this year.
          --limit <n>              Process at most n files (smoke tests).
          --sample <n>             Print parsed passages for n files and exit. Writes nothing.
          --threads <n>            Degree of parallelism. Default: processor count.
          --target-words <n>       Chunk target size. Default: 300.
          --overlap-words <n>      Chunk overlap. Default: 50.
          --shard-size <n>         Articles per Parquet part. Default: 25000.
        """;

    public List<CorpusInput> Inputs { get; } = [];
    public string OutputDirectory { get; private set; } = "";
    public int? YearMin { get; private set; }
    public int? YearMax { get; private set; }
    public int? Limit { get; private set; }
    public int Sample { get; private set; }
    public int Threads { get; private set; } = Environment.ProcessorCount;
    public int TargetWords { get; private set; } = ChunkOptions.Default.TargetWords;
    public int OverlapWords { get; private set; } = ChunkOptions.Default.OverlapWords;
    public int ShardSize { get; private set; } = 25_000;

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
                case "--input":
                    var value = Next();
                    var separator = value.IndexOf('=');
                    if (separator <= 0)
                    {
                        throw new ArgumentException($"--input expects <corpus>=<dir>, got '{value}'.");
                    }
                    options.Inputs.Add(new CorpusInput(value[..separator], value[(separator + 1)..]));
                    break;

                case "--out": options.OutputDirectory = Next(); break;
                case "--year-min": options.YearMin = int.Parse(Next()); break;
                case "--year-max": options.YearMax = int.Parse(Next()); break;
                case "--limit": options.Limit = int.Parse(Next()); break;
                case "--sample": options.Sample = int.Parse(Next()); break;
                case "--threads": options.Threads = int.Parse(Next()); break;
                case "--target-words": options.TargetWords = int.Parse(Next()); break;
                case "--overlap-words": options.OverlapWords = int.Parse(Next()); break;
                case "--shard-size": options.ShardSize = int.Parse(Next()); break;

                default:
                    throw new ArgumentException($"Unrecognised argument '{flag}'.");
            }
        }

        if (options.Inputs.Count == 0) throw new ArgumentException("At least one --input is required.");
        if (options.Sample == 0 && string.IsNullOrWhiteSpace(options.OutputDirectory))
        {
            throw new ArgumentException("--out is required.");
        }

        return options;
    }
}
