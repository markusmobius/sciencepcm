namespace SciencePcm.Core;

/// <summary>
/// One row per article. Deliberately all-scalar: no lists or nested groups, so the
/// Parquet reads cleanly from DuckDB, pyarrow, polars and Parquet.Net alike.
/// Structured authorship and the citation graph come from OpenAlex, not from JATS.
/// </summary>
public sealed class ArticleRow
{
    public string ArticleKey { get; set; } = "";
    public string SourceCorpus { get; set; } = "";
    public string SourcePath { get; set; } = "";

    public string? Pmcid { get; set; }
    public string? Pmid { get; set; }
    public string? Doi { get; set; }

    public string? Title { get; set; }
    public string? Journal { get; set; }
    public string? IssnElectronic { get; set; }
    public string? IssnPrint { get; set; }
    public string? Publisher { get; set; }
    public string? ArticleType { get; set; }

    public int? PubYear { get; set; }
    public int? PubMonth { get; set; }
    public int? PubDay { get; set; }

    public string? LicenseUrl { get; set; }
    public string? LicenseText { get; set; }
    public string? Authors { get; set; }
    public string? Keywords { get; set; }

    public string? Abstract { get; set; }
    public int AbstractWordCount { get; set; }

    public bool IsRetracted { get; set; }
    public bool HasBody { get; set; }
    public int BodyWordCount { get; set; }
    public int SectionCount { get; set; }
    public int ReferenceCount { get; set; }
    public int ChunkCount { get; set; }
}

/// <summary>
/// One row per retrievable passage. Title and PubYear are denormalised so a search hit
/// can be rendered without joining back to <see cref="ArticleRow"/>.
/// </summary>
public sealed class ChunkRow
{
    public string ChunkId { get; set; } = "";
    public string ArticleKey { get; set; } = "";
    public string SourceCorpus { get; set; } = "";

    public string SectionKind { get; set; } = "";
    public string? SectionTitle { get; set; }
    public string? SectionPath { get; set; }

    public int SectionIndex { get; set; }
    public int ChunkIndex { get; set; }

    public string Text { get; set; } = "";
    public int WordCount { get; set; }

    public string? Title { get; set; }
    public int? PubYear { get; set; }
    public bool IsRetracted { get; set; }
}

/// <summary>A paragraph-level unit produced by the parser, before chunking.</summary>
public sealed record ParsedBlock(
    SectionKind Kind,
    string? SectionTitle,
    string? SectionPath,
    int SectionIndex,
    string Text);

public sealed record ParsedArticle(ArticleRow Article, IReadOnlyList<ParsedBlock> Blocks);
