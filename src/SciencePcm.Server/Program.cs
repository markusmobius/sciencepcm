using SciencePcm.Server;

// The configuration binder expects "--key value", so a bare "--gpu" would otherwise
// consume the next switch as its value.
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
    ?? throw new InvalidOperationException("--index <lucene directory> is required.");
var crossEncoder = builder.Configuration["cross-encoder"]
    ?? throw new InvalidOperationException("--cross-encoder <onnx directory> is required.");

var options = new ServerOptions
{
    IndexPath = index,
    CrossEncoderPath = crossEncoder,
    RerankCandidates = builder.Configuration.GetValue("rerank-candidates", 100),
    RerankBatch = builder.Configuration.GetValue("rerank-batch", 32),
    Threads = builder.Configuration.GetValue("threads", 8),
    UseGpu = builder.Configuration.GetValue("gpu", false),
    GpuMemoryLimitBytes = builder.Configuration.GetValue<long>("gpu-mem-limit-gb", 0) * 1024L * 1024 * 1024,
};

var token = builder.Configuration["token"] ?? Environment.GetEnvironmentVariable("SCIENCEPCM_TOKEN");

// Browser clients need CORS, and MCP carries its session in a custom header that must be
// explicitly exposed or the browser will hide it from script.
var corsOrigins = (builder.Configuration["cors-origins"]
        ?? "https://www.mcptest.econlabs.org,http://localhost:5173,http://127.0.0.1:5173")
    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

builder.Services.AddCors(options => options.AddDefaultPolicy(policy => policy
    .WithOrigins(corsOrigins)
    .AllowAnyHeader()
    .AllowAnyMethod()
    .WithExposedHeaders("Mcp-Session-Id")));

builder.Services.AddSingleton(options);
builder.Services.AddSingleton<RetrievalService>();
builder.Services.AddMcpServer().WithHttpTransport().WithToolsFromAssembly();

var app = builder.Build();

app.UseCors();

// A shared-secret check, because this listens on the LAN. Not a substitute for TLS.
if (!string.IsNullOrEmpty(token))
{
    app.Use(async (context, next) =>
    {
        // Preflight carries no Authorization header by design, so rejecting it would
        // stop the browser ever sending the real request.
        if (context.Request.Path.StartsWithSegments("/health")
            || HttpMethods.IsOptions(context.Request.Method))
        {
            await next();
            return;
        }

        var header = context.Request.Headers.Authorization.ToString();
        if (header != $"Bearer {token}")
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
    status = "ok",
    documents = retrieval.DocumentCount,
}));

app.MapMcp("/mcp");

// Touching the service now surfaces a bad index or model path at startup rather than on
// the first request, and pays the model load before anyone is waiting.
var warmup = app.Services.GetRequiredService<RetrievalService>();
Console.WriteLine($"index          : {options.IndexPath} ({warmup.DocumentCount:N0} documents)");
Console.WriteLine($"cross-encoder  : {options.CrossEncoderPath} (gpu={options.UseGpu})");
Console.WriteLine($"rerank depth   : {options.RerankCandidates}");
Console.WriteLine($"auth           : {(string.IsNullOrEmpty(token) ? "OPEN - no token set" : "bearer token required")}");
Console.WriteLine($"mcp endpoint   : /mcp");
Console.WriteLine($"cors origins   : {string.Join(", ", corsOrigins)}");

app.Run();
