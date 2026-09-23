using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Lucene.Net.Search;
using SciencePcm.Index;

namespace OpenAlex.Server;

public sealed class OpenAlexCorpus
{
    public static OpenAlexCorpus Full { get; } = new("openalex", null);

    public string Name { get; }
    public string McpPath => Filter is null ? "/mcp" : $"/mcp/{Name}";
    public int? RequestedIds { get; }
    public Filter? Filter { get; }

    private OpenAlexCorpus(string name, IReadOnlyCollection<string>? ids)
    {
        Name = name;
        RequestedIds = ids?.Count;
        Filter = ids is null ? null : LexicalIndex.CreateKeyFilter(ids);
    }

    public static IReadOnlyList<OpenAlexCorpus> Load(string configPath)
    {
        configPath = Path.GetFullPath(configPath);
        using var stream = File.OpenRead(configPath);
        var configuration = JsonSerializer.Deserialize<CorpusConfiguration>(stream, new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        }) ?? throw new InvalidDataException($"Corpus configuration is null: {configPath}");

        if (configuration.CustomCorpora is null)
            throw new InvalidDataException("customCorpora must be an array, not null.");

        var corpora = new List<OpenAlexCorpus> { Full };
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { Full.Name };
        foreach (var definition in configuration.CustomCorpora)
        {
            if (definition is null || string.IsNullOrWhiteSpace(definition.Name)
                || !Regex.IsMatch(definition.Name, @"\A[a-z0-9][a-z0-9_-]*\z"))
                throw new InvalidDataException("Each corpus needs a name containing only lowercase letters, digits, underscores or hyphens.");

            if (!names.Add(definition.Name))
                throw new InvalidDataException($"Duplicate or reserved corpus name: {definition.Name}");

            if (string.IsNullOrWhiteSpace(definition.IdsFile))
                throw new InvalidDataException($"Corpus '{definition.Name}' needs an idsFile.");

            var idsPath = Path.GetFullPath(definition.IdsFile, Path.GetDirectoryName(configPath)!);
            var ids = new HashSet<string>(StringComparer.Ordinal);
            var lineNumber = 0;
            foreach (var line in File.ReadLines(idsPath))
            {
                lineNumber++;
                if (string.IsNullOrWhiteSpace(line)) continue;
                var id = NormalizeId(line)
                    ?? throw new InvalidDataException($"Invalid OpenAlex work ID at {idsPath}:{lineNumber}");
                ids.Add(id);
            }

            if (ids.Count == 0)
                throw new InvalidDataException($"Corpus '{definition.Name}' has an empty ID list: {idsPath}");

            corpora.Add(new OpenAlexCorpus(definition.Name, ids));
        }

        return corpora.AsReadOnly();
    }

    public static string? NormalizeId(string value)
    {
        value = value.Trim();
        foreach (var prefix in new[] { "https://openalex.org/", "http://openalex.org/" })
        {
            if (!value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) continue;
            value = value[prefix.Length..];
            break;
        }

        return Regex.IsMatch(value, @"\A[Ww][0-9]+\z") ? "https://openalex.org/W" + value[1..] : null;
    }

    private sealed record CorpusConfiguration
    {
        public required List<CorpusDefinition> CustomCorpora { get; init; }
    }

    private sealed record CorpusDefinition
    {
        public required string Name { get; init; }
        public required string IdsFile { get; init; }
    }
}