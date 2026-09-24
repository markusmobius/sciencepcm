using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Logging;
using CustomMcp.Server;
using Xunit;

namespace MitPcm.Tests;

[Collection("MIT corpus")]
public sealed class McpServerTests(MitCorpusFixture fixture) : IAsyncLifetime
{
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    public async Task InitializeAsync()
    {
        _app = CustomServer.Build(["--config", fixture.Config, "--index-root", fixture.IndexRoot,
            "--cross-encoder", fixture.Model, "--token", "test-token", "--threads", "1", "--rerank-batch", "2"], builder =>
        {
            builder.WebHost.UseTestServer();
            builder.Logging.ClearProviders();
        });
        await _app.StartAsync();
        _client = _app.GetTestClient();
        _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "test-token");
        _client.DefaultRequestHeaders.Accept.ParseAdd("application/json");
        _client.DefaultRequestHeaders.Accept.ParseAdd("text/event-stream");
        _client.DefaultRequestHeaders.Add("MCP-Protocol-Version", "2025-11-25");
    }

    [Theory]
    [InlineData("papers")]
    [InlineData("other")]
    public async Task InitializationAndToolsIdentifyTheConfiguredCorpus(string corpus)
    {
        var initialized = await Request("initialize", new
        {
            protocolVersion = "2025-11-25", capabilities = new { },
            clientInfo = new { name = "custom-test", version = "1.0.0" },
        }, corpus);
        Assert.Equal(corpus, initialized.GetProperty("serverInfo").GetProperty("name").GetString());
        var listed = await Request("tools/list", new { }, corpus);
        var tools = listed.GetProperty("tools").EnumerateArray().ToArray();
        Assert.Equal(new[] { "corpus_stats", "get_paper", "get_passage_context", "search_full_text", "search_literature" },
            tools.Select(tool => tool.GetProperty("name").GetString()).OrderBy(name => name, StringComparer.Ordinal));
        Assert.All(tools, tool => Assert.False(tool.GetProperty("description").GetString()!.Contains("2019-2025", StringComparison.Ordinal)));
    }

    [Fact]
    public async Task PaperSearchPreservesAuthorsAndRunsReranking()
    {
        var searched = await Call("search_literature", new { query = "neurons", limit = 5 });
        var hit = Assert.Single(searched.GetProperty("results").EnumerateArray());
        Assert.Equal(MitCorpusFixture.PaperKey, hit.GetProperty("article_key").GetString());
        Assert.Equal("Luis Sa-Couto; Andreas Wichert", hit.GetProperty("authors").GetString());
        Assert.Equal("10.1162/neco_a_01243", hit.GetProperty("doi").GetString());
        Assert.Equal(JsonValueKind.Null, hit.GetProperty("cited_by_count").ValueKind);
        var filtered = await Call("search_literature", new { author = "Wichert" });
        Assert.Equal(1, filtered.GetProperty("returned").GetInt32());
    }

    [Fact]
    public async Task YearFiltersExcludeUndatedRecords()
    {
        var unfiltered = await Call("search_literature", new { author = "Gast" });
        Assert.Equal(MitCorpusFixture.ErratumKey,
            Assert.Single(unfiltered.GetProperty("results").EnumerateArray()).GetProperty("article_key").GetString());
        var upperOnly = await Call("search_literature", new { author = "Gast", yearMax = 2024 });
        Assert.Equal(0, upperOnly.GetProperty("returned").GetInt32());
        var lowerOnly = await Call("search_literature", new { author = "Gast", yearMin = 1 });
        Assert.Equal(0, lowerOnly.GetProperty("returned").GetInt32());
        var dated = await Call("search_literature", new { author = "Wichert", yearMin = 2024, yearMax = 2024 });
        Assert.Equal(1, dated.GetProperty("returned").GetInt32());
    }

    [Fact]
    public async Task LookupReturnsFullAbstractAndTitleOnlyErratum()
    {
        var paper = await Call("get_paper", new { articleKey = MitCorpusFixture.PaperKey });
        Assert.Equal(MitCorpusFixture.Abstract, paper.GetProperty("abstract").GetString());
        var erratum = await Call("get_paper", new { articleKey = MitCorpusFixture.ErratumKey });
        Assert.Equal("", erratum.GetProperty("abstract").GetString());
        Assert.Equal("10.1162/neco_c_01397", erratum.GetProperty("doi").GetString());
        Assert.Equal(0, erratum.GetProperty("year").GetInt32());
        var missing = await Call("get_paper", new { articleKey = "mit:not-here" });
        Assert.Equal("not found", missing.GetProperty("error").GetString());
    }

    [Fact]
    public async Task PassageSearchAndContextUseSharedRetrieval()
    {
        var searched = await Call("search_full_text", new { query = "neurons", section = "Methods", maxPerArticle = 1 });
        var hit = Assert.Single(searched.GetProperty("results").EnumerateArray());
        Assert.Equal("methods", hit.GetProperty("section").GetString());
        Assert.Equal("Luis Sa-Couto; Andreas Wichert", hit.GetProperty("authors").GetString());
        var context = await Call("get_passage_context", new { passageId = hit.GetProperty("passage_id").GetString(), before = 1, after = 1 });
        var passages = context.GetProperty("passages").EnumerateArray().ToArray();
        Assert.Equal(2, passages.Length);
        Assert.All(passages, passage => Assert.Equal(MitCorpusFixture.PaperKey, passage.GetProperty("article_key").GetString()));
    }

    [Fact]
    public async Task StatsReportMeasuredCountsWithoutOtherCorpusClaims()
    {
        var stats = await Call("corpus_stats", new { });
        Assert.Equal(2, stats.GetProperty("documents").GetInt32());
        Assert.Equal(fixture.Result.Chunks, stats.GetProperty("passages").GetInt32());
        Assert.Equal(2, stats.GetProperty("papers_with_passages").GetInt32());
        Assert.False(stats.TryGetProperty("methods_benchmark", out _));
    }

    [Fact]
    public async Task AuthenticationProtectsMcpButAllowsHealthAndPreflight()
    {
        using var anonymous = _app.GetTestClient();
        using var denied = await anonymous.PostAsJsonAsync("/mcp/papers", new { jsonrpc = "2.0", id = 1, method = "tools/list" });
        Assert.Equal(HttpStatusCode.Unauthorized, denied.StatusCode);
        using var health = await anonymous.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, health.StatusCode);
        using var preflight = new HttpRequestMessage(HttpMethod.Options, "/mcp/papers");
        preflight.Headers.Add("Origin", "http://localhost:5173");
        preflight.Headers.Add("Access-Control-Request-Method", "POST");
        using var response = await anonymous.SendAsync(preflight);
        Assert.Equal(HttpStatusCode.NoContent, response.StatusCode);
        Assert.Equal("http://localhost:5173", Assert.Single(response.Headers.GetValues("Access-Control-Allow-Origin")));
    }

    [Fact]
    public async Task StatelessEndpointRejectsSessionReuse()
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "/mcp/papers")
        {
            Content = JsonContent.Create(new { jsonrpc = "2.0", id = 1, method = "tools/list", @params = new { } }),
        };
        request.Headers.Add("MCP-Session-Id", "another-session");
        using var response = await _client.SendAsync(request);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task ConcurrentRequestsShareRerankResourcesWithoutLosingResults()
    {
        await Task.WhenAll(Enumerable.Range(0, 8).Select(async iteration =>
        {
            var corpus = iteration % 2 == 0 ? "papers" : "other";
            var result = await Call("search_literature", new { query = "neurons", limit = 1 }, corpus);
            Assert.Equal(corpus + ":paper-one",
                Assert.Single(result.GetProperty("results").EnumerateArray()).GetProperty("article_key").GetString());
        }));
    }

    [Fact]
    public async Task EveryToolStaysWithinItsNamedCollection()
    {
        var stats = await Call("corpus_stats", new { }, "other");
        Assert.Equal("other", stats.GetProperty("corpus").GetString());
        Assert.Equal(1, stats.GetProperty("documents").GetInt32());
        var missingAuthor = await Call("search_literature", new { author = "Wichert" }, "other");
        Assert.Equal(0, missingAuthor.GetProperty("returned").GetInt32());
        var missingPaper = await Call("get_paper", new { articleKey = MitCorpusFixture.PaperKey }, "other");
        Assert.Equal("not found", missingPaper.GetProperty("error").GetString());
        var reverse = await Call("get_paper", new { articleKey = "other:paper-one" });
        Assert.Equal("not found", reverse.GetProperty("error").GetString());
        var fullText = await Call("search_full_text", new { query = "neurons", maxPerArticle = 5 }, "other");
        Assert.NotEmpty(fullText.GetProperty("results").EnumerateArray());
        Assert.All(fullText.GetProperty("results").EnumerateArray(), hit => Assert.Equal("other:paper-one", hit.GetProperty("article_key").GetString()));
        var context = await Call("get_passage_context", new { passageId = MitCorpusFixture.PaperKey + "#0000" }, "other");
        Assert.Equal("not found", context.GetProperty("error").GetString());
    }

    [Theory]
    [InlineData("/mcp")]
    [InlineData("/mcp/not-configured")]
    public async Task UnboundRoutesCannotFallBackToAnotherCollection(string path)
    {
        using var response = await _client.PostAsJsonAsync(path, new { jsonrpc = "2.0", id = 1, method = "tools/list" });
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    private async Task<JsonElement> Call(string name, object arguments, string corpus = "papers")
    {
        var result = await Request("tools/call", new { name, arguments }, corpus);
        Assert.False(result.TryGetProperty("isError", out var error) && error.GetBoolean(), result.ToString());
        var text = Assert.Single(result.GetProperty("content").EnumerateArray()).GetProperty("text").GetString()!;
        using var json = JsonDocument.Parse(text);
        return json.RootElement.Clone();
    }

    private async Task<JsonElement> Request(string method, object parameters, string corpus = "papers")
    {
        using var response = await _client.PostAsJsonAsync("/mcp/" + corpus, new { jsonrpc = "2.0", id = 1, method, @params = parameters });
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
    }
}