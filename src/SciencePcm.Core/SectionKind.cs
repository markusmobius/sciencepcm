namespace SciencePcm.Core;

/// <summary>
/// IMRaD-normalised section label. Retrieval quality depends on being able to tell a
/// Results claim from a Discussion speculation, so every chunk carries one of these.
/// </summary>
public enum SectionKind
{
    Unknown = 0,
    Abstract,
    Introduction,
    Background,
    Methods,
    Results,
    Discussion,
    Conclusion,
    Limitations,
    FigureCaption,
    TableCaption,
    Supplementary,
    Acknowledgements,
    Other,
}

public static class SectionClassifier
{
    // JATS @sec-type is authoritative when present, but is absent or non-standard in a
    // large fraction of real PMC files, so title matching is the primary path.
    private static readonly (string Needle, SectionKind Kind)[] TitleRules =
    [
        ("materials and methods", SectionKind.Methods),
        ("methods and materials", SectionKind.Methods),
        ("experimental procedure", SectionKind.Methods),
        ("methodology", SectionKind.Methods),
        ("method", SectionKind.Methods),
        ("statistical analysis", SectionKind.Methods),
        ("data analysis", SectionKind.Methods),
        ("study design", SectionKind.Methods),
        ("participants", SectionKind.Methods),
        ("subjects", SectionKind.Methods),
        ("animals", SectionKind.Methods),

        ("results and discussion", SectionKind.Results),
        ("result", SectionKind.Results),
        ("findings", SectionKind.Results),

        ("limitation", SectionKind.Limitations),
        ("future direction", SectionKind.Limitations),

        ("conclusion", SectionKind.Conclusion),
        ("summary", SectionKind.Conclusion),

        ("discussion", SectionKind.Discussion),
        ("interpretation", SectionKind.Discussion),

        ("introduction", SectionKind.Introduction),
        ("background", SectionKind.Background),
        ("related work", SectionKind.Background),

        ("acknowledg", SectionKind.Acknowledgements),
        ("supplementary", SectionKind.Supplementary),
        ("supporting information", SectionKind.Supplementary),
        ("data availability", SectionKind.Supplementary),
        ("author contribution", SectionKind.Acknowledgements),
        ("conflict of interest", SectionKind.Acknowledgements),
        ("competing interest", SectionKind.Acknowledgements),
        ("funding", SectionKind.Acknowledgements),
        ("abbreviation", SectionKind.Other),
    ];

    private static readonly Dictionary<string, SectionKind> SecTypeRules = new(StringComparer.OrdinalIgnoreCase)
    {
        ["intro"] = SectionKind.Introduction,
        ["introduction"] = SectionKind.Introduction,
        ["background"] = SectionKind.Background,
        ["materials|methods"] = SectionKind.Methods,
        ["methods"] = SectionKind.Methods,
        ["materials"] = SectionKind.Methods,
        ["results"] = SectionKind.Results,
        ["results|discussion"] = SectionKind.Results,
        ["discussion"] = SectionKind.Discussion,
        ["conclusions"] = SectionKind.Conclusion,
        ["supplementary-material"] = SectionKind.Supplementary,
        ["abbreviations"] = SectionKind.Other,
        ["acknowledgments"] = SectionKind.Acknowledgements,
        ["acknowledgements"] = SectionKind.Acknowledgements,
        ["COI-statement"] = SectionKind.Acknowledgements,
    };

    public static SectionKind Classify(string? secType, string? title)
    {
        if (!string.IsNullOrWhiteSpace(secType) && SecTypeRules.TryGetValue(secType.Trim(), out var byType))
        {
            return byType;
        }

        if (string.IsNullOrWhiteSpace(title))
        {
            return SectionKind.Unknown;
        }

        var normalised = title.Trim().ToLowerInvariant();

        // Strip a leading numeric label such as "3." or "2.1" before matching.
        var cursor = 0;
        while (cursor < normalised.Length && (char.IsDigit(normalised[cursor]) || normalised[cursor] is '.' or ' '))
        {
            cursor++;
        }
        normalised = normalised[cursor..];

        foreach (var (needle, kind) in TitleRules)
        {
            if (normalised.Contains(needle, StringComparison.Ordinal))
            {
                return kind;
            }
        }

        return SectionKind.Unknown;
    }

    /// <summary>
    /// Sections that carry no retrievable scientific claim. Excluded from chunking so the
    /// index is not padded with funding statements and author contribution boilerplate.
    /// </summary>
    public static bool IsNonEvidential(SectionKind kind) =>
        kind is SectionKind.Acknowledgements or SectionKind.Supplementary;
}
