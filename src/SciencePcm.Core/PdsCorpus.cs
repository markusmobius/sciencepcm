using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace SciencePcm.Core;

public sealed record PdsCorpus
{
    public required string Name { get; init; }
    public required string InputCloudPath { get; init; }
    public string McpPath => $"/mcp/{Name}";

    public static IReadOnlyList<PdsCorpus> Load(string configPath)
    {
        using var stream = File.OpenRead(configPath);
        var configuration = JsonSerializer.Deserialize<CorpusConfiguration>(stream, new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        }) ?? throw new InvalidDataException("PDS corpus configuration cannot be null.");
        if (configuration.CustomCorpora is not { Count: > 0 })
            throw new InvalidDataException("customCorpora must contain at least one PDS collection.");

        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var corpus in configuration.CustomCorpora)
        {
            if (corpus is null) throw new InvalidDataException("A PDS collection cannot be null.");
            ValidateName(corpus.Name);
            if (!names.Add(corpus.Name)) throw new InvalidDataException($"Duplicate corpus name: {corpus.Name}");
            ValidateCloudPath(corpus.InputCloudPath);
        }
        return configuration.CustomCorpora.AsReadOnly();
    }

    public static void ValidateName(string name)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Length > 100 ||
            !Regex.IsMatch(name, @"\A[a-z0-9][a-z0-9_-]*\z", RegexOptions.CultureInvariant))
            throw new InvalidDataException("Corpus names must be 1-100 lowercase letters, digits, underscores or hyphens, beginning with a letter or digit.");
    }

    public static void ValidateCloudPath(string cloudPath)
    {
        if (string.IsNullOrWhiteSpace(cloudPath)) throw new InvalidDataException("inputCloudPath is required.");
        var segments = cloudPath.Split('/');
        if (segments.Length < 2 || !cloudPath.EndsWith(".pds", StringComparison.OrdinalIgnoreCase) ||
            segments.Any(segment => string.IsNullOrWhiteSpace(segment) || segment is "." or ".." ||
                segment != segment.Trim() || segment.Any(character => char.IsControl(character) || "\\:|*?\"<>#".Contains(character))))
            throw new InvalidDataException("inputCloudPath must name one relative cloud PDS file without traversal or reserved characters.");
    }

    private sealed record CorpusConfiguration
    {
        public required List<PdsCorpus> CustomCorpora { get; init; }
    }
}