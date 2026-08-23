using System.Text;
using System.Text.RegularExpressions;
using System.Xml;
using System.Xml.Linq;

namespace SciencePcm.Core;

public static partial class JatsParser
{
    // External DTD resolution is disabled: PMC files declare a JATS DTD we do not need,
    // and resolving it would be both slow and an XXE vector.
    private static readonly XmlReaderSettings ReaderSettings = new()
    {
        DtdProcessing = DtdProcessing.Ignore,
        XmlResolver = null,
        IgnoreComments = true,
        IgnoreProcessingInstructions = true,
        IgnoreWhitespace = false,
    };

    /// <summary>Elements whose text must not enter the retrievable passage stream.</summary>
    private static readonly HashSet<string> SkipInText = new(StringComparer.Ordinal)
    {
        "xref",           // "[12,43]" citation markers
        "fig",            // captured separately as FigureCaption
        "table-wrap",     // captured separately as TableCaption
        "table",
        "disp-formula",
        "inline-formula",
        "tex-math",
        "math",
        "graphic",
        "media",
        "supplementary-material",
        "fn",
        "fn-group",
        "label",
        "object-id",
        "alternatives",
    };

    public static ParsedArticle? Parse(string path, string sourceCorpus)
    {
        XDocument doc;
        using (var stream = File.OpenRead(path))
        using (var reader = XmlReader.Create(stream, ReaderSettings))
        {
            doc = XDocument.Load(reader, LoadOptions.None);
        }

        var article = doc.Root;
        if (article is null || !article.Name.LocalName.Equals("article", StringComparison.Ordinal))
        {
            return null;
        }

        var front = article.El("front");
        var meta = front?.El("article-meta");
        var journalMeta = front?.El("journal-meta");

        var row = new ArticleRow
        {
            SourceCorpus = sourceCorpus,
            SourcePath = path,
            ArticleType = article.Attr("article-type"),
        };

        ReadIdentifiers(meta, row);
        ReadJournal(journalMeta, row);
        ReadTitleAndPeople(meta, row);
        ReadDate(meta, row);
        ReadLicense(meta, row);

        row.IsRetracted =
            (row.ArticleType?.Contains("retract", StringComparison.OrdinalIgnoreCase) ?? false) ||
            article.Descendants().Any(e =>
                e.Name.LocalName == "related-article" &&
                (e.Attr("related-article-type")?.Contains("retract", StringComparison.OrdinalIgnoreCase) ?? false));

        row.ArticleKey = BuildArticleKey(row, path);

        var blocks = new List<ParsedBlock>();
        var sectionIndex = 0;

        var abstractText = ReadAbstract(meta);
        if (!string.IsNullOrWhiteSpace(abstractText))
        {
            row.Abstract = abstractText;
            row.AbstractWordCount = TextUtil.CountWords(abstractText);
            blocks.Add(new ParsedBlock(SectionKind.Abstract, "Abstract", "Abstract", sectionIndex++, abstractText));
        }

        var body = article.El("body");
        if (body is not null)
        {
            CollectBody(body, blocks, ref sectionIndex, row);
            CollectCaptions(body, blocks, ref sectionIndex);
        }

        row.HasBody = blocks.Any(b => b.Kind != SectionKind.Abstract);
        row.BodyWordCount = blocks.Where(b => b.Kind != SectionKind.Abstract).Sum(b => TextUtil.CountWords(b.Text));
        row.SectionCount = sectionIndex;
        row.ReferenceCount = article.El("back")?.Descendants()
            .Count(e => e.Name.LocalName == "ref") ?? 0;

        return new ParsedArticle(row, blocks);
    }

    private static void ReadIdentifiers(XElement? meta, ArticleRow row)
    {
        if (meta is null) return;

        foreach (var id in meta.Els("article-id"))
        {
            var type = id.Attr("pub-id-type");
            var value = id.Value.Trim();
            if (value.Length == 0) continue;

            switch (type)
            {
                case "pmc":
                    row.Pmcid = value.StartsWith("PMC", StringComparison.OrdinalIgnoreCase) ? value.ToUpperInvariant() : "PMC" + value;
                    break;
                case "pmid":
                    row.Pmid = value;
                    break;
                case "doi":
                    row.Doi = value.ToLowerInvariant();
                    break;
            }
        }
    }

    private static void ReadJournal(XElement? journalMeta, ArticleRow row)
    {
        if (journalMeta is null) return;

        row.Journal = journalMeta.El("journal-title-group")?.El("journal-title")?.Value.Trim()
                      ?? journalMeta.El("journal-title")?.Value.Trim();

        foreach (var issn in journalMeta.Els("issn"))
        {
            if (issn.Attr("pub-type") == "epub" || issn.Attr("publication-format") == "electronic")
            {
                row.IssnElectronic = issn.Value.Trim();
            }
            else
            {
                row.IssnPrint = issn.Value.Trim();
            }
        }

        row.Publisher = journalMeta.El("publisher")?.El("publisher-name")?.Value.Trim();
    }

    private static void ReadTitleAndPeople(XElement? meta, ArticleRow row)
    {
        if (meta is null) return;

        var titleElement = meta.El("title-group")?.El("article-title");
        if (titleElement is not null)
        {
            row.Title = TextUtil.Clean(ExtractText(titleElement));
        }

        var authors = new List<string>();
        foreach (var contrib in meta.Descendants().Where(e => e.Name.LocalName == "contrib"))
        {
            if (contrib.Attr("contrib-type") is { } ct && ct != "author") continue;

            var name = contrib.El("name") ?? contrib.El("name-alternatives")?.El("name");
            if (name is null)
            {
                var collab = contrib.El("collab");
                if (collab is not null) authors.Add(collab.Value.Trim());
                continue;
            }

            var surname = name.El("surname")?.Value.Trim();
            var given = name.El("given-names")?.Value.Trim();
            if (string.IsNullOrEmpty(surname)) continue;

            authors.Add(string.IsNullOrEmpty(given) ? surname : $"{surname}, {given}");
        }

        if (authors.Count > 0)
        {
            row.Authors = string.Join("; ", authors);
        }

        var keywords = meta.Els("kwd-group")
            .SelectMany(g => g.Els("kwd"))
            .Select(k => k.Value.Trim())
            .Where(k => k.Length > 0)
            .ToList();

        if (keywords.Count > 0)
        {
            row.Keywords = string.Join("; ", keywords);
        }
    }

    private static void ReadDate(XElement? meta, ArticleRow row)
    {
        if (meta is null) return;

        var dates = meta.Els("pub-date").ToList();
        if (dates.Count == 0) return;

        // Electronic publication is the earliest reliable public-availability date.
        var preferred =
            dates.FirstOrDefault(d => d.Attr("pub-type") == "epub")
            ?? dates.FirstOrDefault(d => d.Attr("date-type") == "pub")
            ?? dates.FirstOrDefault(d => d.Attr("pub-type") == "ppub")
            ?? dates.FirstOrDefault(d => d.Attr("pub-type") == "collection")
            ?? dates.FirstOrDefault(d => d.El("year") is not null);

        if (preferred is null) return;

        if (int.TryParse(preferred.El("year")?.Value.Trim(), out var year)) row.PubYear = year;
        if (int.TryParse(preferred.El("month")?.Value.Trim(), out var month)) row.PubMonth = month;
        if (int.TryParse(preferred.El("day")?.Value.Trim(), out var day)) row.PubDay = day;
    }

    private static void ReadLicense(XElement? meta, ArticleRow row)
    {
        var license = meta?.El("permissions")?.El("license");
        if (license is null) return;

        // Publishers disagree on where the machine-readable URL goes: NISO ALI
        // <ali:license_ref>, an @xlink:href on <license>, or only an <ext-link> in the prose.
        row.LicenseUrl =
            license.Descendants().FirstOrDefault(e => e.Name.LocalName == "license_ref")?.Value.Trim()
            ?? license.Attributes().FirstOrDefault(a => a.Name.LocalName == "href")?.Value
            ?? license.Descendants()
                .Where(e => e.Name.LocalName == "ext-link")
                .Select(e => e.Attributes().FirstOrDefault(a => a.Name.LocalName == "href")?.Value)
                .FirstOrDefault(v => v is not null && v.Contains("creativecommons", StringComparison.OrdinalIgnoreCase));

        var text = license.Els("license-p").FirstOrDefault() ?? license;
        row.LicenseText = TextUtil.Clean(ExtractText(text));
    }

    private static string? ReadAbstract(XElement? meta)
    {
        if (meta is null) return null;

        var candidates = meta.Els("abstract")
            .Where(a => a.Attr("abstract-type") is not ("graphical" or "teaser" or "video" or "media"))
            .ToList();

        var chosen = candidates.FirstOrDefault(a => a.Attr("abstract-type") is null)
                     ?? candidates.FirstOrDefault();

        if (chosen is null) return null;

        // Structured abstracts nest labelled <sec> blocks; flatten them in document order.
        var builder = new StringBuilder();
        foreach (var paragraph in chosen.Descendants().Where(e => e.Name.LocalName == "p"))
        {
            var text = TextUtil.Clean(ExtractText(paragraph));
            if (text.Length == 0) continue;
            if (builder.Length > 0) builder.Append(' ');
            builder.Append(text);
        }

        if (builder.Length == 0)
        {
            builder.Append(TextUtil.Clean(ExtractText(chosen)));
        }

        return builder.Length == 0 ? null : builder.ToString();
    }

    private static void CollectBody(XElement body, List<ParsedBlock> blocks, ref int sectionIndex, ArticleRow row)
    {
        // Paragraphs before the first <sec>, which some publishers use for the opening text.
        var leading = body.Els("p").ToList();
        if (leading.Count > 0)
        {
            var index = sectionIndex++;
            foreach (var p in leading)
            {
                var text = TextUtil.Clean(ExtractText(p));
                if (text.Length > 0)
                {
                    blocks.Add(new ParsedBlock(SectionKind.Introduction, null, null, index, text));
                }
            }
        }

        foreach (var sec in body.Els("sec"))
        {
            WalkSection(sec, SectionKind.Unknown, null, blocks, ref sectionIndex);
        }
    }

    private static void WalkSection(
        XElement sec,
        SectionKind inheritedKind,
        string? parentPath,
        List<ParsedBlock> blocks,
        ref int sectionIndex)
    {
        var title = sec.El("title") is { } t ? TextUtil.Clean(ExtractText(t)) : null;
        var path = string.IsNullOrEmpty(parentPath) ? title : $"{parentPath} > {title}";

        // Anchor to the outermost recognised ancestor so "Methods > Animals" stays Methods.
        var kind = inheritedKind != SectionKind.Unknown
            ? inheritedKind
            : SectionClassifier.Classify(sec.Attr("sec-type"), title);

        var index = sectionIndex++;

        if (!SectionClassifier.IsNonEvidential(kind))
        {
            foreach (var p in sec.Els("p"))
            {
                var text = TextUtil.Clean(ExtractText(p));
                if (text.Length > 0)
                {
                    blocks.Add(new ParsedBlock(kind, title, path, index, text));
                }
            }
        }

        foreach (var child in sec.Els("sec"))
        {
            WalkSection(child, kind, path, blocks, ref sectionIndex);
        }
    }

    private static void CollectCaptions(XElement body, List<ParsedBlock> blocks, ref int sectionIndex)
    {
        // Figure captions in neuroscience routinely state the actual result, so they are
        // indexed rather than discarded with the rest of the float content.
        foreach (var (localName, kind) in new[] { ("fig", SectionKind.FigureCaption), ("table-wrap", SectionKind.TableCaption) })
        {
            foreach (var element in body.Descendants().Where(e => e.Name.LocalName == localName))
            {
                var caption = element.El("caption");
                if (caption is null) continue;

                var text = TextUtil.Clean(ExtractText(caption, allowFloats: true));
                if (text.Length < 20) continue;

                var label = element.El("label")?.Value.Trim();
                blocks.Add(new ParsedBlock(kind, label, label, sectionIndex++, text));
            }
        }
    }

    private static string ExtractText(XElement element, bool allowFloats = false)
    {
        var builder = new StringBuilder();
        AppendText(element, builder, allowFloats);
        return builder.ToString();
    }

    private static void AppendText(XElement element, StringBuilder builder, bool allowFloats)
    {
        foreach (var node in element.Nodes())
        {
            switch (node)
            {
                case XText text:
                    builder.Append(text.Value);
                    break;

                case XElement child:
                    var name = child.Name.LocalName;
                    if (SkipInText.Contains(name) && !(allowFloats && name is "label"))
                    {
                        continue;
                    }
                    AppendText(child, builder, allowFloats);
                    break;
            }
        }
    }

    private static string BuildArticleKey(ArticleRow row, string path)
    {
        if (!string.IsNullOrEmpty(row.Pmcid)) return row.Pmcid;
        if (!string.IsNullOrEmpty(row.Doi)) return "doi:" + row.Doi;
        return "file:" + Path.GetFileNameWithoutExtension(path);
    }

    private static XElement? El(this XElement? parent, string localName) =>
        parent?.Elements().FirstOrDefault(e => e.Name.LocalName == localName);

    private static IEnumerable<XElement> Els(this XElement? parent, string localName) =>
        parent?.Elements().Where(e => e.Name.LocalName == localName) ?? [];

    private static string? Attr(this XElement element, string localName) =>
        element.Attributes().FirstOrDefault(a => a.Name.LocalName == localName)?.Value;
}

public static partial class TextUtil
{
    [GeneratedRegex(@"\[\s*[,;\u2013\u2014-]*\s*\]")]
    private static partial Regex EmptyBrackets();

    [GeneratedRegex(@"\(\s*[,;\u2013\u2014-]*\s*\)")]
    private static partial Regex EmptyParens();

    [GeneratedRegex(@"\s+([,.;:!?])")]
    private static partial Regex SpaceBeforePunctuation();

    [GeneratedRegex(@"\s{2,}")]
    private static partial Regex RepeatedWhitespace();

    /// <summary>
    /// Removes the punctuation debris left behind when citation markers are dropped,
    /// e.g. "shown in ()" or "reported [ , ]", then normalises whitespace.
    /// </summary>
    public static string Clean(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return "";

        var text = raw.Replace('\u00a0', ' ').Replace('\r', ' ').Replace('\n', ' ').Replace('\t', ' ');
        text = EmptyBrackets().Replace(text, "");
        text = EmptyParens().Replace(text, "");
        text = SpaceBeforePunctuation().Replace(text, "$1");
        text = RepeatedWhitespace().Replace(text, " ");
        return text.Trim();
    }

    public static int CountWords(ReadOnlySpan<char> text)
    {
        var count = 0;
        var inWord = false;
        foreach (var c in text)
        {
            if (char.IsWhiteSpace(c)) { inWord = false; }
            else if (!inWord) { inWord = true; count++; }
        }
        return count;
    }
}
