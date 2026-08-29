using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using SciencePcm.Embed;

namespace SciencePcm.Lexical;

/// <summary>
/// What an index was built from, written beside it so a rebuild can be skipped.
///
/// The source fingerprint is over file names and lengths, not contents: reading 134 GB
/// to decide whether to read 134 GB defeats the point, and the ingest rewrites every
/// shard when the digest changes.
/// </summary>
public sealed record IndexStamp(
    int SchemaVersion,
    string Schema,
    bool RequireBody,
    int Documents,
    long Bytes,
    string SourceFingerprint)
{
    public static IndexStamp Describe(
        string glob, string? metadataGlob, CorpusSchema schema, bool requireBody)
    {
        var files = new[] { glob, metadataGlob }
            .Where(pattern => !string.IsNullOrWhiteSpace(pattern))
            .SelectMany(pattern => ParquetTextSource.ExpandGlob(pattern!))
            .Select(path => new FileInfo(path))
            .Where(file => file.Exists)
            .OrderBy(file => file.FullName, StringComparer.Ordinal)
            .ToList();

        var builder = new StringBuilder();
        foreach (var file in files)
        {
            builder.Append(file.Name).Append(':').Append(file.Length).Append('\n');
        }

        var digest = Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString())))[..16];

        return new IndexStamp(
            LexicalIndex.SchemaVersion,
            schema.ToString(),
            requireBody,
            files.Count,
            files.Sum(file => file.Length),
            digest);
    }

    public static IndexStamp? Read(string path)
    {
        if (!File.Exists(path)) return null;
        try
        {
            return JsonSerializer.Deserialize<IndexStamp>(File.ReadAllText(path));
        }
        catch (JsonException)
        {
            return null;
        }
    }

    public static void Write(string path, IndexStamp stamp) =>
        File.WriteAllText(path, JsonSerializer.Serialize(
            stamp, new JsonSerializerOptions { WriteIndented = true }));
}
