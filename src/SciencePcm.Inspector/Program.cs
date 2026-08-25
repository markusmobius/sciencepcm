// Serves the local MCP console. Static files only.
//
// There is deliberately no proxy endpoint: one that forwarded to a caller-supplied URL
// would be a server-side request forgery hole the moment this is exposed publicly.
// Servers the console talks to must send CORS headers instead - see deploy/nginx/.
var builder = WebApplication.CreateBuilder(args);

// launchSettings.json only applies to `dotnet run`, so a published build would
// otherwise fall back to the framework default of :5000.
var url = builder.Configuration["urls"] ?? "http://localhost:5173";
builder.WebHost.UseUrls(url);

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

Console.WriteLine($"MCP console listening on {url}");
Console.WriteLine("Target servers must allow this origin in their CORS policy.");

app.Run();
