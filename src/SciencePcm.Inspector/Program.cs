using System.Text.Json;

// Serves the local MCP console. Static files plus the config the page reads.
//
// There is deliberately no proxy endpoint: one that forwarded to a caller-supplied URL
// would be a server-side request forgery hole the moment this is exposed publicly.
// Servers the console talks to must send CORS headers instead - see deploy/nginx/.
//
// The config file supplies the listen URL, the server list, and per-tool suggested
// inputs. It is optional: without it the console still works against any MCP server,
// just with no presets.
var configPath = args.FirstOrDefault(a => !a.StartsWith('-')) ?? "config.json";
var configJson = File.Exists(configPath) ? File.ReadAllText(configPath) : "{}";

string? ConfigValue(string name)
{
    using var document = JsonDocument.Parse(configJson);
    return document.RootElement.TryGetProperty(name, out var value) ? value.GetString() : null;
}

var builder = WebApplication.CreateBuilder(args);

// launchSettings.json only applies to `dotnet run`, so a published build would
// otherwise fall back to the framework default of :5000.
var url = builder.Configuration["urls"] ?? ConfigValue("urls") ?? "http://localhost:5173";
builder.WebHost.UseUrls(url);

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/config", () => Results.Content(configJson, "application/json"));

Console.WriteLine($"MCP console listening on {url}");
Console.WriteLine($"config: {(File.Exists(configPath) ? Path.GetFullPath(configPath) : "(none - no presets)")}");
Console.WriteLine("Target servers must allow this origin in their CORS policy.");

app.Run();
