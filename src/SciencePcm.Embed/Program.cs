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

        // Tokenizer only: loading the 416 MB ONNX graph would tell us nothing here.
        var tokenizer = TokenizerFactory.Create(
            options.ModelDirectory, options.Tokenizer, maxTokens, options.LowerCase);
        var vocabulary = File.ReadAllLines(Path.Combine(options.ModelDirectory, "vocab.txt"));

        Console.WriteLine($"tokenizer: {TokenizerFactory.Describe(options.ModelDirectory, options.Tokenizer)}");
        Console.WriteLine();

        string Render(IEnumerable<int> ids) => string.Join(
            ' ',
            ids.Select(id => id >= 0 && id < vocabulary.Length ? vocabulary[id] : $"<{id}>"));

        var failures = 0;
        var total = 0;

        foreach (var sample in root.GetProperty("samples").EnumerateArray())
        {
            total++;
            var text = sample.GetProperty("text").GetString()!;
            var expected = sample.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var actual = tokenizer.Encode(text);

            if (expected.AsSpan().SequenceEqual(actual))
            {
                Console.WriteLine($"  OK    {text[..Math.Min(60, text.Length)]}");
                continue;
            }

            failures++;
            Console.WriteLine($"  FAIL  {text}");
            Console.WriteLine($"        python: {Render(expected)}");
            Console.WriteLine($"        csharp: {Render(actual)}");
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
        options.LowerCase,
        options.Tokenizer,
        options.PadMultiple);

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

        ReportTokenLengths(options, sample);

        Console.WriteLine("Warming up ...");

        var embedders = CreateEmbedders(options, out var dimensions);
        try
        {
            // Every session must be warmed, not just the first: an unwarmed session's
            // first inference is slow, and with many workers that dominates the timing.
            Console.WriteLine($"Warming {options.Workers} session(s) ...");
            var warm = sample.Take(Math.Min(options.BatchSize, sample.Count)).ToList();
            Parallel.For(0, options.Workers, slot => embedders[slot].Embed(warm));

            var batches = Chunk(SortByLength(sample, options.SortBatches, options), options.BatchSize).ToList();
            Console.WriteLine($"Running {batches.Count:N0} batches ...");

            var stopwatch = Stopwatch.StartNew();
            var next = -1;
            long paddedTokens = 0;

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
                        Interlocked.Add(ref paddedTokens, embedders[slot].LastBatchTokens);
                    }
                });

            stopwatch.Stop();

            var perSecond = sample.Count / stopwatch.Elapsed.TotalSeconds;
            var tokensPerSecond = paddedTokens / stopwatch.Elapsed.TotalSeconds;
            Console.WriteLine();
            Console.WriteLine($"dimensions      : {dimensions}");
            Console.WriteLine($"texts           : {sample.Count:N0}");
            Console.WriteLine($"elapsed         : {stopwatch.Elapsed.TotalSeconds:F1}s");
            Console.WriteLine($"throughput      : {perSecond:N0} texts/s");
            Console.WriteLine($"padded tokens/s : {tokensPerSecond:N0}");
            Console.WriteLine($"avg padded len  : {(double)paddedTokens / sample.Count:F0} tokens/text");
            Console.WriteLine($"padding waste   : {100.0 * (1.0 - (double)_realTokens / paddedTokens):F1}% of compute");
            Console.WriteLine($"session threads : {Math.Clamp(options.Sessions, 1, options.Workers) * options.IntraOpThreads}"
                              + $"  ({Math.Clamp(options.Sessions, 1, options.Workers)} session(s) x {options.IntraOpThreads} intra-op)");
            Console.WriteLine($"worker threads  : {options.Workers}");
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
            DisposeSessions();
        }

        return 0;
    }

    /// <summary>
    /// Cost scales with tokens, not documents, so the truncation limit should be chosen
    /// from the real length distribution rather than guessed.
    /// </summary>
    private static void ReportTokenLengths(Options options, List<string> sample)
    {
        var tokenizer = TokenizerFactory.Create(
            options.ModelDirectory, options.Tokenizer, options.MaxTokens, options.LowerCase);

        var lengths = new int[sample.Count];
        Parallel.For(0, sample.Count, i => lengths[i] = tokenizer.Encode(sample[i]).Length);
        Array.Sort(lengths);

        int Percentile(double p) => lengths[Math.Clamp((int)(p * lengths.Length), 0, lengths.Length - 1)];

        Console.WriteLine();
        Console.WriteLine($"token lengths (truncated at --max-tokens {options.MaxTokens}):");
        Console.WriteLine($"  mean {lengths.Average():F0}   p50 {Percentile(0.50)}   p90 {Percentile(0.90)}   " +
                          $"p95 {Percentile(0.95)}   p99 {Percentile(0.99)}   max {lengths[^1]}");
        _realTokens = lengths.Sum(l => (long)l);

        foreach (var limit in new[] { 128, 192, 256, 384, 512 })
        {
            var cost = lengths.Sum(l => (long)Math.Min(l, limit));
            var full = lengths.Sum(l => (long)l);
            var truncated = lengths.Count(l => l > limit);
            Console.WriteLine($"  --max-tokens {limit,3}: {100.0 * cost / full,5:F1}% of compute, " +
                              $"{100.0 * truncated / lengths.Length,5:F1}% of texts truncated");
        }
        Console.WriteLine();
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
            var buffer = new List<TextRecord>(Math.Max(options.BatchSize, options.SortBuffer));
            var seen = 0;

            var sortTokenizer = options.SortBatches
                ? TokenizerFactory.Create(
                    options.ModelDirectory, options.Tokenizer, options.MaxTokens, options.LowerCase)
                : null;

            async Task DrainAsync()
            {
                if (buffer.Count == 0) return;

                // Length-sort within the buffer only, so the stream is never fully materialised.
                var ordered = buffer;
                if (sortTokenizer is not null)
                {
                    var lengths = new int[buffer.Count];
                    Parallel.For(0, buffer.Count, i => lengths[i] = sortTokenizer.Encode(buffer[i].Text).Length);
                    var order = Enumerable.Range(0, buffer.Count).ToArray();
                    Array.Sort(order, (a, b) => lengths[a].CompareTo(lengths[b]));
                    ordered = order.Select(i => buffer[i]).ToList();
                }

                for (var i = 0; i < ordered.Count; i += options.BatchSize)
                {
                    var count = Math.Min(options.BatchSize, ordered.Count - i);
                    await batchChannel.Writer.WriteAsync(ordered.GetRange(i, count));
                }

                buffer = new List<TextRecord>(Math.Max(options.BatchSize, options.SortBuffer));
            }

            await foreach (var record in ParquetTextSource.ReadAsync(
                               options.Input, options.Schema, options.IncludeTitle))
            {
                buffer.Add(record);
                seen++;

                if (buffer.Count >= Math.Max(options.BatchSize, options.SortBuffer))
                {
                    await DrainAsync();
                }

                if (options.Limit is { } limit && seen >= limit) break;
            }

            await DrainAsync();
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
        DisposeSessions();

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
        var sessionCount = Math.Clamp(options.Sessions, 1, options.Workers);

        _sessions = new Microsoft.ML.OnnxRuntime.InferenceSession[sessionCount];
        for (var i = 0; i < sessionCount; i++)
        {
            _sessions[i] = TextEmbedder.CreateSession(
                options.ModelDirectory, options.IntraOpThreads, options.Gpu, options.GpuDevice,
                options.GpuMemLimitGb > 0 ? (long)(options.GpuMemLimitGb * 1024L * 1024L * 1024L) : 0);
        }

        var embedders = new TextEmbedder[options.Workers];
        for (var i = 0; i < options.Workers; i++)
        {
            embedders[i] = new TextEmbedder(embedderOptions, _sessions[i % sessionCount]);
        }

        Console.WriteLine($"sessions={sessionCount} (shared across {options.Workers} workers)"
                          + (options.Gpu ? $"  provider=CUDA:{options.GpuDevice}" : "  provider=CPU"));
        dimensions = embedders[0].Dimensions;
        return embedders;
    }

    private static Microsoft.ML.OnnxRuntime.InferenceSession[] _sessions = [];

    // Set by ReportTokenLengths so the benchmark can show how much of the padded
    // compute was real content.
    private static long _realTokens;

    private static void DisposeSessions()
    {
        foreach (var session in _sessions) session.Dispose();
        _sessions = [];
    }

    private static IEnumerable<List<string>> Chunk(List<string> items, int size)
    {
        for (var i = 0; i < items.Count; i += size)
        {
            yield return items.GetRange(i, Math.Min(size, items.Count - i));
        }
    }

    /// <summary>
    /// Groups similar-length texts together. A batch pads to its longest member, so mixing
    /// a 60-token abstract with a 400-token one wastes most of the forward pass on padding.
    /// Sorting on the real token count rather than character length matters: characters are
    /// a weak proxy, and the difference measured about 458 padded tokens/text versus 265.
    /// The extra tokenisation pass is microseconds against a millisecond forward pass.
    /// </summary>
    private static List<string> SortByLength(List<string> texts, bool enabled, Options options)
    {
        if (!enabled) return texts;

        var tokenizer = TokenizerFactory.Create(
            options.ModelDirectory, options.Tokenizer, options.MaxTokens, options.LowerCase);

        var lengths = new int[texts.Count];
        Parallel.For(0, texts.Count, i => lengths[i] = tokenizer.Encode(texts[i]).Length);

        var order = Enumerable.Range(0, texts.Count).ToArray();
        Array.Sort(order, (a, b) => lengths[a].CompareTo(lengths[b]));
        return order.Select(i => texts[i]).ToList();
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
          --tokenizer fast|ml    fast = FastBertTokenizer, reads HuggingFace tokenizer.json.
                                 ml   = Microsoft.ML.Tokenizers, reads vocab.txt.
                                 Default: fast

        Throughput:
          --workers <n>          Concurrent embedding threads. Default: 16
          --sessions <n>         ONNX sessions shared by those workers. Each session holds
                                 its own copy of the weights (~440 MB), so keep this low.
                                 Default: 1
          --intra-threads <n>    Threads per session. Default: 8
          --batch <n>            Texts per forward pass. Default: 64
          --shard-size <n>       Vectors per output shard. Default: 250000
          --pad-multiple <n>     Round padded sequence length up to a multiple of this.
                                 Keeps the GPU allocator from fragmenting over a long run.
                                 1 disables. Default: 64
          --sort-buffer <n>      Texts buffered and length-sorted before batching, which
                                 cuts padding waste. 0 disables. Default: 65536
          --no-sort              Disable length-sorted batching.
          --limit <n>            Stop after n texts.
          --benchmark            Measure throughput and project full-corpus time.
          --benchmark-texts <n>  Sample size for --benchmark. Default: 5000

        GPU (requires: dotnet build -c Release -p:UseGpu=true):
          --gpu                  Use the CUDA execution provider.
          --gpu-device <n>       CUDA device index. Default: 0
          --gpu-mem-limit <gb>   Cap GPU memory for this process, so another workload on
                                 the same card is not starved. 0 = unlimited. Default: 0

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
    public TokenizerKind Tokenizer { get; private set; } = TokenizerKind.Fast;
    public int Workers { get; private set; } = 16;
    public int Sessions { get; private set; } = 1;
    public int IntraOpThreads { get; private set; } = 8;
    public int BatchSize { get; private set; } = 64;
    public int ShardSize { get; private set; } = 250_000;
    public int PadMultiple { get; private set; } = 64;
    public int SortBuffer { get; private set; } = 65_536;
    public bool SortBatches { get; private set; } = true;
    public int? Limit { get; private set; }
    public bool Benchmark { get; private set; }
    public bool Gpu { get; private set; }
    public int GpuDevice { get; private set; }
    public double GpuMemLimitGb { get; private set; }
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
                case "--tokenizer":
                    options.Tokenizer = Next().ToLowerInvariant() switch
                    {
                        "fast" => TokenizerKind.Fast,
                        "ml" => TokenizerKind.MlNet,
                        var other => throw new ArgumentException($"Unknown tokenizer '{other}'."),
                    };
                    break;
                case "--workers": options.Workers = int.Parse(Next()); break;
                case "--sessions": options.Sessions = int.Parse(Next()); break;
                case "--intra-threads": options.IntraOpThreads = int.Parse(Next()); break;
                case "--batch": options.BatchSize = int.Parse(Next()); break;
                case "--shard-size": options.ShardSize = int.Parse(Next()); break;
                case "--pad-multiple": options.PadMultiple = int.Parse(Next()); break;
                case "--sort-buffer": options.SortBuffer = int.Parse(Next()); break;
                case "--no-sort": options.SortBatches = false; break;
                case "--limit": options.Limit = int.Parse(Next()); break;
                case "--benchmark": options.Benchmark = true; break;
                case "--gpu": options.Gpu = true; break;
                case "--gpu-device": options.GpuDevice = int.Parse(Next()); break;
                case "--gpu-mem-limit": options.GpuMemLimitGb = double.Parse(Next()); break;
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
