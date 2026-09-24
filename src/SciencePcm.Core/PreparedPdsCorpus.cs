using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace SciencePcm.Core;

public sealed record PreparedPdsCorpus
{
    public int SchemaVersion { get; init; } = 1;
    public required string Name { get; init; }
    public required string InputCloudPath { get; init; }
    public required string InputSha256 { get; init; }
    public required string Generation { get; init; }
    public required int Documents { get; init; }
    public required int Passages { get; init; }

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    public string IndexDirectory(string indexRoot) => Path.Combine(Path.GetFullPath(indexRoot), Name, Generation);

    public static PreparedPdsCorpus Load(PdsCorpus corpus, string indexRoot)
    {
        PdsCorpus.ValidateName(corpus.Name);
        using var stream = File.OpenRead(Path.Combine(indexRoot, corpus.Name, "current.json"));
        var prepared = JsonSerializer.Deserialize<PreparedPdsCorpus>(stream, JsonOptions)
            ?? throw new InvalidDataException($"Missing preparation state for '{corpus.Name}'.");
        prepared.Validate();
        if (prepared.Name != corpus.Name || prepared.InputCloudPath != corpus.InputCloudPath)
            throw new InvalidDataException($"Configuration changed for '{corpus.Name}'; run CustomMcp prepare before serving.");
        return prepared;
    }

    public async Task PublishAsync(string indexRoot)
    {
        Validate();
        var directory = Path.Combine(Path.GetFullPath(indexRoot), Name);
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "current.json");
        var content = JsonSerializer.Serialize(this, JsonOptions);
        if (File.Exists(path) && await File.ReadAllTextAsync(path) == content) return;
        var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            await File.WriteAllTextAsync(temporary, content);
            File.Move(temporary, path, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
        }
    }

    private void Validate()
    {
        PdsCorpus.ValidateName(Name);
        PdsCorpus.ValidateCloudPath(InputCloudPath);
        if (SchemaVersion != 1 || Documents < 1 || Passages < 0 ||
            !Regex.IsMatch(InputSha256 ?? "", @"\A[0-9a-f]{64}\z") ||
            !Regex.IsMatch(Generation ?? "", @"\A[0-9a-f]{64}\z"))
            throw new InvalidDataException($"Invalid preparation state for '{Name}'.");
    }
}