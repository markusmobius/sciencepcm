using ModelContextProtocol.Server;
using SciencePcm.Core;
using SciencePcm.Server;

namespace CustomMcp.Server;

public static class CustomServer
{
    public static WebApplication Build(string[] args, Action<WebApplicationBuilder>? configureBuilder = null)
    {
        var builder = WebApplication.CreateBuilder(ExpandFlags(args));
        configureBuilder?.Invoke(builder);
        if (string.IsNullOrWhiteSpace(builder.Configuration["urls"]))
            builder.WebHost.UseUrls("http://127.0.0.1:8083");

        string Required(string name) => builder.Configuration[name] is { Length: > 0 } value
            ? value : throw new ArgumentException($"--{name} <directory> is required.");
        var configPath = builder.Configuration["config"] ?? Path.Combine(AppContext.BaseDirectory, "custom_mcps", "pds.json");
        var corpora = PdsCorpus.Load(configPath);
        var indexRoot = Path.GetFullPath(Required("index-root"));
        var options = new ServerOptions
        {
            IndexPath = indexRoot,
            CrossEncoderPath = Required("cross-encoder"),
            RerankCandidates = builder.Configuration.GetValue("rerank-candidates", 100),
            RerankBatch = builder.Configuration.GetValue("rerank-batch", 32),
            MaxConcurrentReranks = builder.Configuration.GetValue("rerank-concurrency", 1),
            Threads = builder.Configuration.GetValue("threads", 8),
            UseGpu = builder.Configuration.GetValue("gpu", false),
            GpuMemoryLimitBytes = checked(builder.Configuration.GetValue<long>("gpu-mem-limit-gb", 0) * 1024L * 1024 * 1024),
        };
        if (options.RerankCandidates < 1 || options.RerankBatch < 1 || options.MaxConcurrentReranks < 1 ||
            options.Threads < 1 || options.GpuMemoryLimitBytes < 0)
            throw new ArgumentException("Rerank sizes, concurrency, and threads must be positive; GPU memory must be nonnegative.");

        var token = builder.Configuration["token"] ?? Environment.GetEnvironmentVariable("CUSTOMMCP_TOKEN");
        var origins = (builder.Configuration["cors-origins"]
            ?? "https://www.mcptest.econlabs.org,http://localhost:5173,http://127.0.0.1:5173")
            .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        builder.Services.AddCors(cors => cors.AddDefaultPolicy(policy => policy.WithOrigins(origins)
            .AllowAnyHeader().AllowAnyMethod().WithExposedHeaders("Mcp-Session-Id")));
        builder.Services.AddSingleton(options);
        builder.Services.AddSingleton<SharedReranker>();
        builder.Services.AddSingleton(services => new CustomCorpusRegistry(corpora, indexRoot, options,
            services.GetRequiredService<SharedReranker>()));
        builder.Services.AddMcpServer().WithHttpTransport(CustomEndpoint.ConfigureTransport);

        var app = builder.Build();
        try
        {
            app.UseCors();
            if (!string.IsNullOrEmpty(token))
            {
                app.Use(async (context, next) =>
                {
                    if (context.Request.Path == "/health" || HttpMethods.IsOptions(context.Request.Method))
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
            var registry = app.Services.GetRequiredService<CustomCorpusRegistry>();
            app.MapGet("/health", () => Results.Ok(new
            {
                service = "custommcp",
                status = "ok",
                collections = registry.Collections.Select(collection => new
                {
                    name = collection.Definition.Name,
                    endpoint = collection.Definition.McpPath,
                    documents = collection.Retrieval.DocumentCount,
                    passages = collection.Retrieval.PassageCount,
                }),
            }));
            foreach (var collection in registry.Collections)
            {
                new CustomEndpoint(collection).Map(app);
                Console.WriteLine($"{collection.Definition.McpPath}: {collection.Retrieval.DocumentCount:N0} papers, {collection.Retrieval.PassageCount:N0} passages");
            }
            Console.WriteLine($"service        : CustomMcp ({corpora.Count} collections)");
            Console.WriteLine($"cross-encoder  : {options.CrossEncoderPath} (one shared session, gpu={options.UseGpu})");
            Console.WriteLine($"rerank slots   : {options.MaxConcurrentReranks} shared across collections");
            Console.WriteLine($"auth           : {(string.IsNullOrEmpty(token) ? "OPEN - no CUSTOMMCP_TOKEN set" : "bearer token required")}");
            return app;
        }
        catch
        {
            app.DisposeAsync().AsTask().GetAwaiter().GetResult();
            throw;
        }
    }

    private static string[] ExpandFlags(string[] args)
    {
        var expanded = new List<string>();
        for (var index = 0; index < args.Length; index++)
        {
            expanded.Add(args[index]);
            if (args[index] == "--gpu" && (index + 1 == args.Length || args[index + 1].StartsWith('-')))
                expanded.Add("true");
        }
        return expanded.ToArray();
    }
}