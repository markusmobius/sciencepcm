using ModelContextProtocol.AspNetCore;
using ModelContextProtocol.Server;

namespace OpenAlex.Server;

public sealed class OpenAlexEndpoint
{
    private readonly OpenAlexCorpus _corpus;
    private readonly McpServerPrimitiveCollection<McpServerTool> _tools = [];

    public OpenAlexEndpoint(OpenAlexCorpus corpus, IEnumerable<McpServerTool> tools)
    {
        _corpus = corpus;
        foreach (var tool in tools) _tools.Add(tool);
    }

    public void Map(IEndpointRouteBuilder endpoints) =>
        endpoints.MapMcp(_corpus.McpPath).WithMetadata(this);

    public static void ConfigureTransport(HttpServerTransportOptions options)
    {
        options.SessionMode = HttpServerSessionMode.Stateless;
        options.ConfigureSessionOptions = (context, serverOptions, cancellationToken) =>
        {
            var endpoint = context.GetEndpoint()?.Metadata.GetMetadata<OpenAlexEndpoint>()
                ?? throw new InvalidOperationException("An MCP endpoint has no corpus binding.");
            serverOptions.ServerInfo = new()
            {
                Name = endpoint._corpus.Name,
                Version = serverOptions.ServerInfo?.Version ?? "1.0.0",
            };
            serverOptions.ServerInstructions = endpoint._corpus.Filter is null
                ? "This endpoint searches the full local OpenAlex snapshot."
                : $"This endpoint only serves the '{endpoint._corpus.Name}' ID allowlist. Absent results do not imply absence from OpenAlex overall.";
            serverOptions.ToolCollection = endpoint._tools;
            return Task.CompletedTask;
        };
    }
}