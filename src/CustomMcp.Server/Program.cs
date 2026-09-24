using CustomMcp.Server;

if (args.Contains("--help") || args.Contains("-h"))
{
    Console.WriteLine("""
        CustomMcp: named PDS collections using shared SciencePCM retrieval.

        --config <file>              Default: custom_mcps/pds.json beside the executable.
        --index-root <directory>     Required prepared collection indexes and current.json files.
        --cross-encoder <directory>  Required ONNX cross-encoder model/tokenizer, loaded once.
        --urls <URL>                 Default: http://127.0.0.1:8083.
        --token <value>              Alternatively set CUSTOMMCP_TOKEN. Unset means no authentication.
        --rerank-candidates <n>      Default: 100.
        --rerank-batch <n>           Default: 32.
        --rerank-concurrency <n>     Default: 1 shared slot across all collections.
        --threads <n>               Default: 8.
        --gpu                       Use a GPU-enabled build and installed CUDA dependencies.
        --gpu-mem-limit-gb <n>       Default: 0 (runtime default).
        --cors-origins <list>        Comma-separated allowed browser origins.

        Endpoints: /mcp/<configured-name> and /health. No unscoped /mcp endpoint.
        Run CustomMcp.Ingest prepare first. Restart after changing configuration or preparing new data.
        """);
    return;
}

var app = CustomServer.Build(args);
app.Run();