using SciencePcm.Server;

var builder = WebApplication.CreateBuilder(ExpandFlags(args, "gpu"));

static string[] ExpandFlags(string[] args, params string[] flags)
{
    var expanded = new List<string>(args.Length + flags.Length);
    for (var i = 0; i < args.Length; i++)
    {
        expanded.Add(args[i]);
        if (!flags.Contains(args[i].TrimStart('-'))) continue;

        var hasValue = i + 1 < args.Length && !args[i + 1].StartsWith('-');
        if (!hasValue) expanded.Add("true");
    }
    return [.. expanded];
}

var index = builder.Configuration["index"]
    ?? throw new InvalidOperationException("--index <OpenAlex Lucene directory> is required.");
var crossEncoder = builder.Configuration["cross-encoder"]
    ?? throw new InvalidOperationException("--cross-encoder <ONNX directory> is required.");

var options = new ServerOptions
{
    IndexPath = index,
    CrossEncoderPath = crossEncoder,
    RerankCandidates = builder.Configuration.GetValue("rerank-candidates", 100),
    RerankBatch = builder.Configuration.GetValue("rerank-batch", 32),
    Threads = builder.Configuration.GetValue("threads", 8),
    ParallelSearch = builder.Configuration.GetValue("parallel-search", true),
    MaxDocFreqRatio = builder.Configuration.GetValue("max-doc-freq-ratio", 0.0),
    ExcludeWorkTypes = (builder.Configuration["exclude-types"] ?? "peer-review,dataset,paratext")
        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries),
    CitationPriorWeight = builder.Configuration.GetValue("citation-prior", 0.5),
    Bm25B = builder.Configuration.GetValue("bm25-b", 0.75f),
    UseGpu = builder.Configuration.GetValue("gpu", false),
    GpuMemoryLimitBytes = builder.Configuration.GetValue<long>("gpu-mem-limit-gb", 0) * 1024L * 1024 * 1024,
};

var token = builder.Configuration["token"] ?? Environment.GetEnvironmentVariable("OPENALEX_TOKEN");
var corsOrigins = (builder.Configuration["cors-origins"]
    ?? "https://www.mcptest.econlabs.org,http://localhost:5173,http://127.0.0.1:5173")
    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

builder.Services.AddCors(cors => cors.AddDefaultPolicy(policy => policy
    .WithOrigins(corsOrigins)
    .AllowAnyHeader()
    .AllowAnyMethod()
    .WithExposedHeaders("Mcp-Session-Id")));
builder.Services.AddSingleton(options);
builder.Services.AddSingleton<RetrievalService>();
builder.Services.AddMcpServer().WithHttpTransport().WithToolsFromAssembly();

var app = builder.Build();
app.UseCors();

if (!string.IsNullOrEmpty(token))
{
    app.Use(async (context, next) =>
    {
        if (context.Request.Path.StartsWithSegments("/health") || HttpMethods.IsOptions(context.Request.Method))
        {
            await next();
            return;
        }

        if (context.Request.Headers.Authorization.ToString() != $"Bearer {token}")
        {
            context.Response.StatusCode = StatusCodes.Status401Unauthorized;
            await context.Response.WriteAsync("unauthorized");
            return;
        }
        await next();
    });
}

app.MapGet("/health", (RetrievalService retrieval) => Results.Ok(new
{
    service = "OpenAlex",
    status = "ok",
    abstracts = retrieval.DocumentCount,
}));
app.MapMcp("/mcp");

var warmup = app.Services.GetRequiredService<RetrievalService>();
Console.WriteLine("service        : OpenAlex MCP");
Console.WriteLine($"abstract index : {options.IndexPath} ({warmup.DocumentCount:N0} documents)");
Console.WriteLine($"cross-encoder  : {options.CrossEncoderPath} (gpu={options.UseGpu})");
Console.WriteLine($"rerank depth   : {options.RerankCandidates}");
Console.WriteLine($"excluded types : {(options.ExcludeWorkTypes.Count == 0 ? "none" : string.Join(", ", options.ExcludeWorkTypes))}");
Console.WriteLine($"citation prior : {options.CitationPriorWeight:0.##} (BM25 x up to {1 + options.CitationPriorWeight:0.##})");
Console.WriteLine($"bm25 b         : {options.Bm25B:0.##} (length normalisation)");
Console.WriteLine($"auth           : {(string.IsNullOrEmpty(token) ? "OPEN - no OPENALEX_TOKEN set" : "bearer token required")}");
Console.WriteLine("mcp endpoint   : /mcp");

app.Run();