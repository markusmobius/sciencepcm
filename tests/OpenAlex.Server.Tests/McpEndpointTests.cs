using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Server;
using Xunit;

namespace OpenAlex.Server.Tests;

public sealed class McpEndpointTests : IAsyncLifetime
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "openalex-http-tests-" + Guid.NewGuid());
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    public async Task InitializeAsync()
    {
        Directory.CreateDirectory(_directory);
        File.WriteAllText(Path.Combine(_directory, "first.txt"), "W1");
        File.WriteAllText(Path.Combine(_directory, "second.txt"), "W2\nW3");
        var configPath = Path.Combine(_directory, "config.json");
        File.WriteAllText(configPath, """
            {"customCorpora":[{"name":"first","idsFile":"first.txt"},{"name":"second","idsFile":"second.txt"}]}
            """);

        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddMcpServer().WithHttpTransport(OpenAlexEndpoint.ConfigureTransport);
        _app = builder.Build();

        foreach (var corpus in OpenAlexCorpus.Load(configPath))
        {
            var scope = McpServerTool.Create(() => corpus.Name, new() { Name = "scope" });
            new OpenAlexEndpoint(corpus, [scope]).Map(_app);
        }

        await _app.StartAsync();
        _client = _app.GetTestClient();
        _client.DefaultRequestHeaders.Accept.ParseAdd("application/json");
        _client.DefaultRequestHeaders.Accept.ParseAdd("text/event-stream");
        _client.DefaultRequestHeaders.Add("MCP-Protocol-Version", "2025-11-25");
    }

    [Theory]
    [InlineData("/mcp", "openalex")]
    [InlineData("/mcp/first", "first")]
    [InlineData("/mcp/second", "second")]
    public async Task EachEndpointHasItsOwnIdentityAndTools(string path, string name)
    {
        var initialized = await Request(path, "initialize", new
        {
            protocolVersion = "2025-11-25",
            capabilities = new { },
            clientInfo = new { name = "corpus-test", version = "1.0.0" },
        });
        Assert.Equal(name, initialized.GetProperty("serverInfo").GetProperty("name").GetString());
        Assert.True(initialized.GetProperty("capabilities").TryGetProperty("tools", out _));

        var listed = await Request(path, "tools/list", new { });
        Assert.Equal("scope", Assert.Single(listed.GetProperty("tools").EnumerateArray()).GetProperty("name").GetString());
        Assert.Equal(name, await ReadScope(path));
    }

    [Fact]
    public async Task ConcurrentRequestsCannotChangeAnotherEndpointsScope()
    {
        var scopes = new[] { ("/mcp", "openalex"), ("/mcp/first", "first"), ("/mcp/second", "second") };
        await Task.WhenAll(Enumerable.Range(0, 30).Select(async iteration =>
        {
            var (path, expected) = scopes[iteration % scopes.Length];
            Assert.Equal(expected, await ReadScope(path));
        }));
    }

    [Fact]
    public async Task SessionHeaderCannotOverrideTheEndpointScope()
    {
        Assert.Equal("openalex", await ReadScope("/mcp"));
        foreach (var path in new[] { "/mcp/first", "/mcp/second" })
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, path)
            {
                Content = JsonContent.Create(new
                {
                    jsonrpc = "2.0", id = 1, method = "tools/call",
                    @params = new { name = "scope", arguments = new { } },
                }),
            };
            request.Headers.Add("MCP-Session-Id", "full-corpus-session");
            using var response = await _client.SendAsync(request);
            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        }
        Assert.Equal("first", await ReadScope("/mcp/first"));
        Assert.Equal("second", await ReadScope("/mcp/second"));
    }

    [Fact]
    public async Task UnknownCorpusDoesNotFallBackToFullSearch()
    {
        using var response = await _client.PostAsJsonAsync("/mcp/not-configured", new
        {
            jsonrpc = "2.0", id = 1, method = "tools/list", @params = new { },
        });
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public void BuildOutputContainsTheDefaultMitConfiguration()
    {
        var corpora = OpenAlexCorpus.Load(Path.Combine(AppContext.BaseDirectory, "custom_mcps", "config.json"));
        Assert.Equal("/mcp", corpora[0].McpPath);
        var mit = Assert.Single(corpora, corpus => corpus.Name == "openalexmit");
        Assert.Equal("/mcp/openalexmit", mit.McpPath);
        Assert.True(mit.RequestedIds > 0);
    }

    private async Task<string?> ReadScope(string path)
    {
        var result = await Request(path, "tools/call", new { name = "scope", arguments = new { } });
        return Assert.Single(result.GetProperty("content").EnumerateArray()).GetProperty("text").GetString();
    }

    private async Task<JsonElement> Request(string path, string method, object parameters)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path)
        {
            Content = JsonContent.Create(new { jsonrpc = "2.0", id = 1, method, @params = parameters }),
        };
        using var response = await _client.SendAsync(request);
        var body = await response.Content.ReadAsStringAsync();
        Assert.True(response.IsSuccessStatusCode, $"HTTP {(int)response.StatusCode}: {body}");
        Assert.False(response.Headers.Contains("MCP-Session-Id"));

        if (response.Content.Headers.ContentType?.MediaType == "text/event-stream")
            body = body.Split('\n').Last(line => line.StartsWith("data:", StringComparison.Ordinal))[5..].Trim();

        using var json = JsonDocument.Parse(body);
        Assert.False(json.RootElement.TryGetProperty("error", out _), body);
        return json.RootElement.GetProperty("result").Clone();
    }

    public async Task DisposeAsync()
    {
        _client?.Dispose();
        if (_app is not null) await _app.DisposeAsync();
        Directory.Delete(_directory, recursive: true);
    }
}