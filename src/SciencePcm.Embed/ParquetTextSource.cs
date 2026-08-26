using Parquet;
using Parquet.Serialization;
using Parquet.Serialization.Attributes;

namespace SciencePcm.Embed;

public sealed record TextRecord(string Id, string Text);

/// <summary>Title and body kept apart, plus the fields the server displays and filters on.</summary>
public sealed record ArticleRecord(
    string Id,
    string ArticleKey,
    string Title,
    string Body,
    int Year,
    string Pmid,
    string Section,
    bool IsRetracted,
    BibliographicMetadata? Metadata = null);

public sealed record BibliographicMetadata(
    string PublicationDate,
    string Doi,
    string Authors,
    string Institutions,
    string Journal,
    string Issn,
    string Language,
    string WorkType,
    int CitedByCount,
    string Volume,
    string Issue,
    string FirstPage,
    string LastPage,
    string Topics,
    string Keywords);

public enum CorpusSchema
{
    /// <summary>OpenAlex-neuroscience-abstracts: openalex_id, title, abstract.</summary>
    Abstracts,

    /// <summary>OpenAlex.Ingest v2 output with bibliographic matching metadata.</summary>
    OpenAlex,

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
    // These differ by source: the abstracts Parquet marks openalex_id required, while
    // Parquet.Net writes every string column as optional, so our own chunk output is all
    // nullable even where the writing type was non-nullable.
    private sealed class AbstractRow
    {
        [ParquetRequired]
        public string openalex_id { get; set; } = "";
        public string? title { get; set; }
        public string? @abstract { get; set; }
    }

    private sealed class ChunkTextRow
    {
        public string? ChunkId { get; set; }
        public string? Title { get; set; }
        public string? Text { get; set; }
    }

    // Separate DTO from AbstractRow: asking for columns the embedder never needs would
    // slow every embedding run down for the sake of the server.
    private sealed class ArticleMetaRow
    {
        [ParquetRequired]
        public string openalex_id { get; set; } = "";
        public string? title { get; set; }
        public string? @abstract { get; set; }
        public int? publication_year { get; set; }
        public string? pmid { get; set; }
    }

    private sealed class OpenAlexMetaRow
    {
        [ParquetRequired]
        public string openalex_id { get; set; } = "";
        public string? title { get; set; }
        public string? @abstract { get; set; }
        public string? publication_date { get; set; }
        public int? publication_year { get; set; }
        public string? pmid { get; set; }
        public string? doi { get; set; }
        public string? authors { get; set; }
        public string? institutions { get; set; }
        public string? journal { get; set; }
        public string? issn { get; set; }
        public string? language { get; set; }
        public string? type { get; set; }
        public int cited_by_count { get; set; }
        public string? volume { get; set; }
        public string? issue { get; set; }
        public string? first_page { get; set; }
        public string? last_page { get; set; }
        public string? topics { get; set; }
        public string? keywords { get; set; }
        public bool is_retracted { get; set; }
    }

    // IsRetracted is a non-nullable bool in ChunkRow, so it is required here too;
    // Parquet.Net throws if the nullability does not match the file exactly.
    private sealed class ChunkMetaRow
    {
        public string? ChunkId { get; set; }
        public string? ArticleKey { get; set; }
        public string? SectionKind { get; set; }
        public string? Text { get; set; }
        public string? Title { get; set; }
        public int? PubYear { get; set; }
        public bool IsRetracted { get; set; }
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

            if (schema is CorpusSchema.Abstracts or CorpusSchema.OpenAlex)
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

    /// <summary>Title, body and metadata kept separate, for building the served index.</summary>
    public static async IAsyncEnumerable<ArticleRecord> ReadArticlesAsync(string glob, CorpusSchema schema)
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
                foreach (var row in (await ParquetSerializer.DeserializeAsync<ArticleMetaRow>(stream)).Data)
                {
                    if (string.IsNullOrEmpty(row.openalex_id) || string.IsNullOrWhiteSpace(row.@abstract)) continue;
                    yield return new ArticleRecord(
                        row.openalex_id,
                        row.openalex_id,
                        row.title ?? "",
                        row.@abstract!,
                        row.publication_year ?? 0,
                        row.pmid ?? "",
                        Section: "",
                        IsRetracted: false);
                }
            }
            else if (schema == CorpusSchema.OpenAlex)
            {
                foreach (var row in (await ParquetSerializer.DeserializeAsync<OpenAlexMetaRow>(stream)).Data)
                {
                    if (string.IsNullOrEmpty(row.openalex_id) || string.IsNullOrWhiteSpace(row.@abstract)) continue;
                    yield return new ArticleRecord(
                        row.openalex_id,
                        row.openalex_id,
                        row.title ?? "",
                        row.@abstract!,
                        row.publication_year ?? 0,
                        row.pmid ?? "",
                        Section: "",
                        row.is_retracted,
                        new BibliographicMetadata(
                            row.publication_date ?? "",
                            row.doi ?? "",
                            row.authors ?? "",
                            row.institutions ?? "",
                            row.journal ?? "",
                            row.issn ?? "",
                            row.language ?? "",
                            row.type ?? "",
                            row.cited_by_count,
                            row.volume ?? "",
                            row.issue ?? "",
                            row.first_page ?? "",
                            row.last_page ?? "",
                            row.topics ?? "",
                            row.keywords ?? ""));
                }
            }
            else
            {
                foreach (var row in (await ParquetSerializer.DeserializeAsync<ChunkMetaRow>(stream)).Data)
                {
                    if (string.IsNullOrEmpty(row.ChunkId) || string.IsNullOrWhiteSpace(row.Text)) continue;
                    yield return new ArticleRecord(
                        row.ChunkId!,
                        row.ArticleKey ?? row.ChunkId!,
                        row.Title ?? "",
                        row.Text!,
                        row.PubYear ?? 0,
                        Pmid: "",
                        row.SectionKind ?? "",
                        row.IsRetracted);
                }
            }
        }
    }
}
