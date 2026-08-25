// Serves the local MCP console. Static files only: the page talks to the MCP server
// directly from the browser, which is why the server allows this origin in CORS.
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

var url = app.Configuration["urls"] ?? "http://localhost:5173";
Console.WriteLine($"MCP console: {url}");
Console.WriteLine("The MCP server must allow this origin: --cors-origins http://localhost:5173");

app.Run();
