using Parquet;
using Parquet.Serialization;
using Parquet.Serialization.Attributes;

namespace SciencePcm.Embed;

public sealed record TextRecord(string Id, string Text);

public enum CorpusSchema
{
    /// <summary>OpenAlex-neuroscience-abstracts: openalex_id, title, abstract.</summary>
    Abstracts,

    /// <summary>SciencePcm.Ingest output: ChunkId, Title, Text.</summary>
    Chunks,
}

/// <summary>
/// Parquet.Net 6.x exposes only typed deserialisation, so each supported corpus needs a
/// DTO whose property names match its column names exactly.
/// </summary>
public static class ParquetTextSource
{
    // Nullability must mirror the file's definition levels exactly or Parquet.Net throws.
    private sealed class AbstractRow
    {
        [ParquetRequired]
        public string openalex_id { get; set; } = "";
        public string? title { get; set; }
        public string? @abstract { get; set; }
    }

    private sealed class ChunkTextRow
    {
        [ParquetRequired]
        public string ChunkId { get; set; } = "";
        public string? Title { get; set; }
        [ParquetRequired]
        public string Text { get; set; } = "";
    }

    public static IEnumerable<string> ExpandGlob(string glob)
    {
        var directory = Path.GetDirectoryName(glob);
        var pattern = Path.GetFileName(glob);

        if (string.IsNullOrEmpty(directory) || !pattern.Contains('*'))
        {
            return File.Exists(glob) ? [glob] : [];
        }

        return Directory.EnumerateFiles(directory, pattern).OrderBy(p => p, StringComparer.Ordinal);
    }

    public static async IAsyncEnumerable<TextRecord> ReadAsync(string glob, CorpusSchema schema, bool includeTitle)
    {
        var files = ExpandGlob(glob).ToList();
        if (files.Count == 0)
        {
            throw new FileNotFoundException($"No Parquet files matched '{glob}'.");
        }

        foreach (var path in files)
        {
            await using var stream = File.OpenRead(path);

            if (schema == CorpusSchema.Abstracts)
            {
                foreach (var row in (await ParquetSerializer.DeserializeAsync<AbstractRow>(stream)).Data)
                {
                    var record = Compose(row.openalex_id, row.title, row.@abstract, includeTitle);
                    if (record is not null) yield return record;
                }
            }
            else
            {
                foreach (var row in (await ParquetSerializer.DeserializeAsync<ChunkTextRow>(stream)).Data)
                {
                    var record = Compose(row.ChunkId, row.Title, row.Text, includeTitle);
                    if (record is not null) yield return record;
                }
            }
        }
    }

    private static TextRecord? Compose(string? id, string? title, string? body, bool includeTitle)
    {
        if (string.IsNullOrEmpty(id) || string.IsNullOrWhiteSpace(body)) return null;

        var text = includeTitle && !string.IsNullOrWhiteSpace(title)
            ? title + ". " + body
            : body;

        return new TextRecord(id, text);
    }
}
