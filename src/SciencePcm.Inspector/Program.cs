// Serves the local MCP console and proxies its requests.
//
// The proxy exists because most MCP servers send no CORS headers, so a browser refuses
// to call them from localhost even when the server itself is reachable. Requests go to
// this origin instead, and the forwarding happens server-side where CORS does not apply.
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddHttpClient("mcp").ConfigureHttpClient(c => c.Timeout = TimeSpan.FromMinutes(10));

// launchSettings.json only applies to `dotnet run`, so a published build would
// otherwise fall back to the framework default of :5000.
var url = builder.Configuration["urls"] ?? "http://localhost:5173";
builder.WebHost.UseUrls(url);

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/proxy", async (HttpRequest request, HttpResponse response, IHttpClientFactory factory) =>
{
    var target = request.Headers["X-Target-Url"].ToString();
    if (string.IsNullOrWhiteSpace(target) || !Uri.TryCreate(target, UriKind.Absolute, out var uri))
    {
        response.StatusCode = StatusCodes.Status400BadRequest;
        await response.WriteAsync("X-Target-Url must be an absolute URL.");
        return;
    }

    using var reader = new StreamReader(request.Body);
    var body = await reader.ReadToEndAsync();

    using var forwarded = new HttpRequestMessage(HttpMethod.Post, uri)
    {
        Content = new StringContent(body, System.Text.Encoding.UTF8, "application/json"),
    };

    foreach (var name in new[] { "Authorization", "Accept", "Mcp-Session-Id", "x-api-key" })
    {
        if (request.Headers.TryGetValue(name, out var value))
        {
            forwarded.Headers.TryAddWithoutValidation(name, value.ToString());
        }
    }

    var client = factory.CreateClient("mcp");
    using var result = await client.SendAsync(forwarded, HttpCompletionOption.ResponseHeadersRead);

    response.StatusCode = (int)result.StatusCode;
    if (result.Content.Headers.ContentType is not null)
    {
        response.ContentType = result.Content.Headers.ContentType.ToString();
    }
    if (result.Headers.TryGetValues("Mcp-Session-Id", out var session))
    {
        response.Headers["Mcp-Session-Id"] = session.First();
    }

    await response.WriteAsync(await result.Content.ReadAsStringAsync());
});

var url = app.Configuration["urls"] ?? "http://localhost:5173";
Console.WriteLine($"MCP console listening on {url}");
Console.WriteLine("Proxy mode forwards to any server; direct mode needs CORS on that server.");

app.Run();
