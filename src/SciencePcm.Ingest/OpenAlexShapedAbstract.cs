using Parquet.Serialization.Attributes;
using SciencePcm.Core;

namespace SciencePcm.Ingest;

/// <summary>
/// A JATS article projected into the column shape of the OpenAlex abstracts Parquet, so
/// that <c>SciencePcm.Index --schema abstracts</c> and the server read a JATS-only corpus
/// with no changes. Column names, casing and nullability must match that file exactly;
/// Parquet.Net throws when definition levels disagree.
/// </summary>
internal sealed class OpenAlexShapedAbstract
{
    // The id column, not an OpenAlex identifier: a JATS corpus has none, and the server
    // uses this purely as the article key.
    [ParquetRequired]
    public string openalex_id { get; set; } = "";

    public string? doi { get; set; }
    public string? title { get; set; }
    public string? @abstract { get; set; }
    public string? publication_date { get; set; }
    public int? publication_year { get; set; }
    public string? pmid { get; set; }
    public string? pmcid { get; set; }
    public string? language { get; set; }
    public string? work_type { get; set; }
    public long? cited_by_count { get; set; }
    public bool? is_oa { get; set; }
    public string? best_oa_landing_page_url { get; set; }
    public string? best_oa_pdf_url { get; set; }
    public string? best_oa_license { get; set; }
    public string? primary_landing_page_url { get; set; }
    public string? primary_pdf_url { get; set; }
    public string? primary_source_name { get; set; }

    public static OpenAlexShapedAbstract From(ArticleRow row)
    {
        var landingPage = string.IsNullOrEmpty(row.Doi) ? null : "https://doi.org/" + row.Doi;

        return new OpenAlexShapedAbstract
        {
            openalex_id = row.ArticleKey,
            doi = row.Doi,
            title = row.Title,
            @abstract = row.Abstract,
            publication_date = FormatDate(row),
            publication_year = row.PubYear,
            pmid = row.Pmid,
            pmcid = row.Pmcid,
            language = null,
            work_type = row.ArticleType,
            // No citation graph in JATS; left null rather than reported as zero.
            cited_by_count = null,
            // A missing licence statement is not evidence of a closed article, so only a
            // present one is asserted.
            is_oa = row.LicenseUrl is null ? null : true,
            best_oa_landing_page_url = row.LicenseUrl is null ? null : landingPage,
            best_oa_pdf_url = null,
            best_oa_license = row.LicenseUrl,
            primary_landing_page_url = landingPage,
            primary_pdf_url = null,
            primary_source_name = row.Journal,
        };
    }

    private static string? FormatDate(ArticleRow row)
    {
        if (row.PubYear is not { } year) return null;
        var month = row.PubMonth is >= 1 and <= 12 ? row.PubMonth.Value : 1;
        var day = row.PubDay is >= 1 and <= 31 ? row.PubDay.Value : 1;
        return $"{year:D4}-{month:D2}-{day:D2}";
    }
}
