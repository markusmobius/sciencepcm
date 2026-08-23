using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

namespace SciencePcm.Embed;

internal static class Program
{
    private static async Task<int> Main(string[] args)
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

        return options.Benchmark
            ? await BenchmarkAsync(options)
            : options.VerifyTokenizer is not null
                ? VerifyTokenizer(options)
                : await EmbedAsync(options);
    }

    /// <summary>
    /// Replays token ids captured from the Python tokenizer at export time. A mismatch
    /// means index-time and query-time tokenisation disagree, which silently wrecks
    /// retrieval quality without producing any error.
    /// </summary>
    private static int VerifyTokenizer(Options options)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(options.VerifyTokenizer!));
        var root = document.RootElement;
        var maxTokens = root.GetProperty("max_tokens").GetInt32();

        using var embedder = new TextEmbedder(ToEmbedderOptions(options) with { MaxTokens = maxTokens });

        var failures = 0;
        var total = 0;

        foreach (var sample in root.GetProperty("samples").EnumerateArray())
        {
            total++;
            var text = sample.GetProperty("text").GetString()!;
            var expected = sample.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var actual = embedder.Tokenize(text).ToArray();

            if (expected.AsSpan().SequenceEqual(actual))
            {
                Console.WriteLine($"  OK    {text[..Math.Min(60, text.Length)]}");
                continue;
            }

            failures++;
            Console.WriteLine($"  FAIL  {text}");
            Console.WriteLine($"        python: {string.Join(' ', expected)}");
            Console.WriteLine($"        csharp: {string.Join(' ', actual)}");
        }

        Console.WriteLine();
        Console.WriteLine($"{total - failures}/{total} probes matched.");
        if (failures > 0)
        {
            Console.Error.WriteLine("Tokenizer mismatch. Do NOT build an index until this is resolved;");
            Console.Error.WriteLine("check --cased/--no-cased and that vocab.txt came from the same model.");
        }

        return failures == 0 ? 0 : 3;
    }

    private static EmbedderOptions ToEmbedderOptions(Options options) => new(
        options.ModelDirectory,
        options.IntraOpThreads,
        options.MaxTokens,
        options.Pooling,
        options.Normalize,
        options.LowerCase);

    private static async Task<int> BenchmarkAsync(Options options)
    {
        Console.WriteLine($"Loading sample of {options.BenchmarkTexts:N0} texts ...");
        var sample = new List<string>(options.BenchmarkTexts);
        await foreach (var record in ParquetTextSource.ReadAsync(
                           options.Input, options.Schema, options.IncludeTitle))
        {
            sample.Add(record.Text);
            if (sample.Count >= options.BenchmarkTexts) break;
        }

        if (sample.Count == 0)
        {
            Console.Error.WriteLine("No texts read; check --input and column names.");
            return 1;
        }

        Console.WriteLine($"workers={options.Workers}  intra-threads={options.IntraOpThreads}  batch={options.BatchSize}");
        Console.WriteLine("Warming up ...");

        var embedders = CreateEmbedders(options, out var dimensions);
        try
        {
            embedders[0].Embed(sample.Take(Math.Min(options.BatchSize, sample.Count)).ToList());

            var batches = Chunk(sample, options.BatchSize).ToList();
            var stopwatch = Stopwatch.StartNew();
            var next = -1;

            await Parallel.ForEachAsync(
                Enumerable.Range(0, options.Workers),
                async (slot, _) =>
                {
                    await Task.Yield();
                    while (true)
                    {
                        var index = Interlocked.Increment(ref next);
                        if (index >= batches.Count) break;
                        embedders[slot].Embed(batches[index]);
                    }
                });

            stopwatch.Stop();

            var perSecond = sample.Count / stopwatch.Elapsed.TotalSeconds;
            Console.WriteLine();
            Console.WriteLine($"dimensions      : {dimensions}");
            Console.WriteLine($"texts           : {sample.Count:N0}");
            Console.WriteLine($"elapsed         : {stopwatch.Elapsed.TotalSeconds:F1}s");
            Console.WriteLine($"throughput      : {perSecond:N0} texts/s");
            Console.WriteLine();
            foreach (var corpus in new[] { 5_298_493L, 6_028_782L })
            {
                var hours = corpus / perSecond / 3600;
                Console.WriteLine($"projected {corpus,10:N0} texts: {hours,6:F1} h");
            }
        }
        finally
        {
            foreach (var embedder in embedders) embedder.Dispose();
        }

        return 0;
    }

    private static async Task<int> EmbedAsync(Options options)
    {
        Directory.CreateDirectory(options.OutputDirectory);

        var embedders = CreateEmbedders(options, out var dimensions);
        Console.WriteLine($"Model loaded: {dimensions} dimensions, pooling={options.Pooling}, normalize={options.Normalize}");
        Console.WriteLine($"workers={options.Workers}  intra-threads={options.IntraOpThreads}  batch={options.BatchSize}");

        var batchChannel = Channel.CreateBounded<List<TextRecord>>(new BoundedChannelOptions(options.Workers * 4)
        {
            SingleWriter = true,
            FullMode = BoundedChannelFullMode.Wait,
        });

        var resultChannel = Channel.CreateBounded<(string[] Ids, float[][] Vectors)>(
            new BoundedChannelOptions(options.Workers * 4) { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });

        var stopwatch = Stopwatch.StartNew();
        var embedded = 0L;

        var producer = Task.Run(async () =>
        {
            var batch = new List<TextRecord>(options.BatchSize);
            var seen = 0;

            await foreach (var record in ParquetTextSource.ReadAsync(
                               options.Input, options.Schema, options.IncludeTitle))
            {
                batch.Add(record);
                seen++;

                if (batch.Count >= options.BatchSize)
                {
                    await batchChannel.Writer.WriteAsync(batch);
                    batch = new List<TextRecord>(options.BatchSize);
                }

                if (options.Limit is { } limit && seen >= limit) break;
            }

            if (batch.Count > 0) await batchChannel.Writer.WriteAsync(batch);
            batchChannel.Writer.Complete();
        });

        var workers = Enumerable.Range(0, options.Workers).Select(slot => Task.Run(async () =>
        {
            await foreach (var batch in batchChannel.Reader.ReadAllAsync())
            {
                var texts = batch.Select(r => r.Text).ToList();
                var vectors = embedders[slot].Embed(texts);
                var ids = batch.Select(r => r.Id).ToArray();

                await resultChannel.Writer.WriteAsync((ids, vectors));

                var done = Interlocked.Add(ref embedded, batch.Count);
                if (done % 50_000 < options.BatchSize)
                {
                    var rate = done / stopwatch.Elapsed.TotalSeconds;
                    Console.WriteLine($"  {done:N0} embedded  ({rate:N0}/s)");
                }
            }
        })).ToArray();

        var writer = Task.Run(() => WriteAsync(resultChannel.Reader, options, dimensions));

        await producer;
        await Task.WhenAll(workers);
        resultChannel.Writer.Complete();
        var (shards, written) = await writer;

        stopwatch.Stop();

        var report = new
        {
            created_at = DateTimeOffset.UtcNow,
            model = embedders[0].Describe(),
            input = options.Input,
            schema = options.Schema.ToString(),
            include_title = options.IncludeTitle,
            vectors_written = written,
            shards,
            dimensions,
            vector_format = "little-endian float32, row-major, no header",
            batch_size = options.BatchSize,
            workers = options.Workers,
            intra_op_threads = options.IntraOpThreads,
            elapsed_seconds = stopwatch.Elapsed.TotalSeconds,
            texts_per_second = written / stopwatch.Elapsed.TotalSeconds,
        };

        await File.WriteAllTextAsync(
            Path.Combine(options.OutputDirectory, "embed-report.json"),
            JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));

        foreach (var embedder in embedders) embedder.Dispose();

        Console.WriteLine();
        Console.WriteLine($"Vectors written : {written:N0} x {dimensions}");
        Console.WriteLine($"Shards          : {shards}");
        Console.WriteLine($"Elapsed         : {stopwatch.Elapsed}");
        return 0;
    }

    private static async Task<(int Shards, long Written)> WriteAsync(
        ChannelReader<(string[] Ids, float[][] Vectors)> reader,
        Options options,
        int dimensions)
    {
        var shard = 0;
        long written = 0;
        var inShard = 0;

        FileStream? vectorStream = null;
        StreamWriter? idStream = null;

        void OpenShard()
        {
            vectorStream = File.Create(Path.Combine(options.OutputDirectory, $"vectors-part-{shard:D4}.f32"));
            idStream = new StreamWriter(Path.Combine(options.OutputDirectory, $"ids-part-{shard:D4}.txt"), false, Encoding.UTF8);
            inShard = 0;
        }

        OpenShard();

        await foreach (var (ids, vectors) in reader.ReadAllAsync())
        {
            for (var i = 0; i < ids.Length; i++)
            {
                vectorStream!.Write(MemoryMarshal.AsBytes<float>(vectors[i]));
                idStream!.WriteLine(ids[i]);
                written++;
                inShard++;
            }

            if (inShard >= options.ShardSize)
            {
                await vectorStream!.FlushAsync();
                vectorStream.Dispose();
                idStream!.Dispose();
                shard++;
                OpenShard();
            }
        }

        await vectorStream!.FlushAsync();
        vectorStream.Dispose();
        idStream!.Dispose();

        return (shard + 1, written);
    }

    private static TextEmbedder[] CreateEmbedders(Options options, out int dimensions)
    {
        var embedderOptions = ToEmbedderOptions(options);
        var embedders = new TextEmbedder[options.Workers];
        for (var i = 0; i < options.Workers; i++)
        {
            embedders[i] = new TextEmbedder(embedderOptions);
        }
        dimensions = embedders[0].Dimensions;
        return embedders;
    }

    private static IEnumerable<List<string>> Chunk(List<string> items, int size)
    {
        for (var i = 0; i < items.Count; i += size)
        {
            yield return items.GetRange(i, Math.Min(size, items.Count - i));
        }
    }
}

internal sealed class Options
{
    public const string Usage = """
        Usage:
          SciencePcm.Embed --model <dir> --input <parquet-glob> --out <dir> [options]
          SciencePcm.Embed --model <dir> --input <parquet-glob> --benchmark [options]

        Required:
          --model <dir>          Directory holding model.onnx and vocab.txt.
          --input <glob>         Parquet glob, e.g. "...\data\part-*.parquet".
          --out <dir>            Output directory (not needed with --benchmark).

        Columns:
          --schema abstracts|chunks   Input layout. Default: abstracts
                                      abstracts -> openalex_id, title, abstract
                                      chunks    -> ChunkId, Title, Text
          --no-title                  Do not prepend the title to the text.

        Model behaviour (must match at query time):
          --max-tokens <n>       Default: 512
          --pooling cls|mean     Default: cls (MedCPT and most BERT retrievers)
          --no-normalize         Skip L2 normalisation.
          --cased                Disable lower-casing (default is uncased).

        Throughput:
          --workers <n>          Concurrent ONNX sessions. Default: 16
          --intra-threads <n>    Threads per session. Default: 8
          --batch <n>            Texts per forward pass. Default: 64
          --shard-size <n>       Vectors per output shard. Default: 250000
          --limit <n>            Stop after n texts.
          --benchmark            Measure throughput and project full-corpus time.
          --benchmark-texts <n>  Sample size for --benchmark. Default: 5000

        Validation:
          --verify-tokenizer <tokenizer-parity.json>
                                 Compare C# tokenisation against the Python export.
                                 Run this before any full index build.
        """;

    public string ModelDirectory { get; private set; } = "";
    public string Input { get; private set; } = "";
    public string OutputDirectory { get; private set; } = "";
    public CorpusSchema Schema { get; private set; } = CorpusSchema.Abstracts;
    public bool IncludeTitle { get; private set; } = true;
    public int MaxTokens { get; private set; } = 512;
    public PoolingMode Pooling { get; private set; } = PoolingMode.Cls;
    public bool Normalize { get; private set; } = true;
    public bool LowerCase { get; private set; } = true;
    public int Workers { get; private set; } = 16;
    public int IntraOpThreads { get; private set; } = 8;
    public int BatchSize { get; private set; } = 64;
    public int ShardSize { get; private set; } = 250_000;
    public int? Limit { get; private set; }
    public bool Benchmark { get; private set; }
    public int BenchmarkTexts { get; private set; } = 5_000;
    public string? VerifyTokenizer { get; private set; }

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
                case "--model": options.ModelDirectory = Next(); break;
                case "--input": options.Input = Next(); break;
                case "--out": options.OutputDirectory = Next(); break;
                case "--schema":
                    options.Schema = Next().ToLowerInvariant() switch
                    {
                        "abstracts" => CorpusSchema.Abstracts,
                        "chunks" => CorpusSchema.Chunks,
                        var other => throw new ArgumentException($"Unknown schema '{other}'."),
                    };
                    break;
                case "--no-title": options.IncludeTitle = false; break;
                case "--max-tokens": options.MaxTokens = int.Parse(Next()); break;
                case "--pooling":
                    options.Pooling = Next().ToLowerInvariant() switch
                    {
                        "cls" => PoolingMode.Cls,
                        "mean" => PoolingMode.MeanOverAttended,
                        var other => throw new ArgumentException($"Unknown pooling '{other}'."),
                    };
                    break;
                case "--no-normalize": options.Normalize = false; break;
                case "--cased": options.LowerCase = false; break;
                case "--workers": options.Workers = int.Parse(Next()); break;
                case "--intra-threads": options.IntraOpThreads = int.Parse(Next()); break;
                case "--batch": options.BatchSize = int.Parse(Next()); break;
                case "--shard-size": options.ShardSize = int.Parse(Next()); break;
                case "--limit": options.Limit = int.Parse(Next()); break;
                case "--benchmark": options.Benchmark = true; break;
                case "--benchmark-texts": options.BenchmarkTexts = int.Parse(Next()); break;
                case "--verify-tokenizer": options.VerifyTokenizer = Next(); break;
                default: throw new ArgumentException($"Unrecognised argument '{flag}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(options.ModelDirectory)) throw new ArgumentException("--model is required.");
        if (options.VerifyTokenizer is not null) return options;
        if (string.IsNullOrWhiteSpace(options.Input)) throw new ArgumentException("--input is required.");
        if (!options.Benchmark && string.IsNullOrWhiteSpace(options.OutputDirectory))
        {
            throw new ArgumentException("--out is required unless --benchmark is set.");
        }

        return options;
    }
}
