using SciencePcm.Server;

var builder = WebApplication.CreateBuilder(args);

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

builder.Services.AddSingleton(options);
builder.Services.AddSingleton<RetrievalService>();
builder.Services.AddMcpServer().WithHttpTransport().WithToolsFromAssembly();

var app = builder.Build();

// A shared-secret check, because this listens on the LAN. Not a substitute for TLS.
if (!string.IsNullOrEmpty(token))
{
    app.Use(async (context, next) =>
    {
        if (context.Request.Path.StartsWithSegments("/health"))
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

app.Run();
