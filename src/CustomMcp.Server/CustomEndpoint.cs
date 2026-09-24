using ModelContextProtocol.AspNetCore;
using ModelContextProtocol.Server;

namespace CustomMcp.Server;

public sealed class CustomEndpoint
{
    private readonly CustomCollection _collection;
    private readonly McpServerPrimitiveCollection<McpServerTool> _tools = [];

    public CustomEndpoint(CustomCollection collection)
    {
        _collection = collection;
        var tools = new CustomTools(collection);
        _tools.Add(McpServerTool.Create(tools.SearchLiterature));
        _tools.Add(McpServerTool.Create(tools.SearchFullText));
        _tools.Add(McpServerTool.Create(tools.GetPaper));
        _tools.Add(McpServerTool.Create(tools.GetPassageContext));
        _tools.Add(McpServerTool.Create(tools.CorpusStats));
    }

    public void Map(IEndpointRouteBuilder endpoints) =>
        endpoints.MapMcp(_collection.Definition.McpPath).WithMetadata(this);

    public static void ConfigureTransport(HttpServerTransportOptions options)
    {
        options.SessionMode = HttpServerSessionMode.Stateless;
        options.ConfigureSessionOptions = (context, serverOptions, cancellationToken) =>
        {
            var endpoint = context.GetEndpoint()?.Metadata.GetMetadata<CustomEndpoint>()
                ?? throw new InvalidOperationException("The CustomMcp endpoint has no collection binding.");
            var name = endpoint._collection.Definition.Name;
            serverOptions.ServerInfo = new() { Name = name, Version = "1.0.0" };
            serverOptions.ServerInstructions = $"CustomMcp serves only the '{name}' PDS collection on this endpoint. " +
                "Use search_literature for discovery, search_full_text for methods and findings, get_paper for abstracts, " +
                "and get_passage_context for surrounding passages. No tool searches another collection. " +
                "Missing dates are unknown, not inferred; year filters exclude undated papers. " +
                "Call corpus_stats for measured counts and limitations. Absence here does not imply absence from the wider literature.";
            serverOptions.ToolCollection = endpoint._tools;
            return Task.CompletedTask;
        };
    }
}