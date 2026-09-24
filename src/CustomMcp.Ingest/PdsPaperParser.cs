using System.Globalization;
using System.Text.Json;
using SciencePcm.Core;

namespace CustomMcp.Ingest;

public static class PdsPaperParser
{
    public static ParsedArticle Parse(JsonElement paper, string paperKey, string sourcePath, string corpusName)
    {
        PdsCorpus.ValidateName(corpusName);
        if (paper.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("A paper must be a JSON object.");
        if (string.IsNullOrWhiteSpace(paperKey))
            throw new InvalidDataException("A paper key is required.");

        var metadata = paper.GetProperty("metadata");
        var title = ReadString(metadata, "title");
        if (string.IsNullOrWhiteSpace(title))
            throw new InvalidDataException("A paper title is required.");

        var abstractText = string.Join(" ", ReadText(Property(paper, "abstract")));
        var authors = string.Join("; ", ReadText(Property(metadata, "authors")));
        var row = new ArticleRow
        {
            ArticleKey = corpusName + ":" + paperKey,
            SourceCorpus = corpusName,
            SourcePath = sourcePath,
            Title = title,
            Authors = authors.Length == 0 ? null : authors,
            Journal = ReadString(metadata, "journal"),
            Doi = ReadString(metadata, "doi"),
            Abstract = abstractText.Length == 0 ? null : abstractText,
            AbstractWordCount = TextUtil.CountWords(abstractText),
            ReferenceCount = Property(metadata, "bibliography") is { ValueKind: JsonValueKind.Array } bibliography
                ? bibliography.GetArrayLength() : 0,
        };
        var date = ReadString(metadata, "publish date");
        if (DateOnly.TryParseExact(date, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var parsedDate))
        {
            row.PubYear = parsedDate.Year;
            row.PubMonth = parsedDate.Month;
            row.PubDay = parsedDate.Day;
        }
        else if (int.TryParse(date, NumberStyles.None, CultureInfo.InvariantCulture, out var year) && year is >= 1 and <= 9999)
        {
            row.PubYear = year;
        }

        var blocks = new List<ParsedBlock>();
        var sectionIndex = 0;
        if (row.Abstract is not null)
            blocks.Add(new ParsedBlock(SectionKind.Abstract, "Abstract", "Abstract", sectionIndex++, row.Abstract));

        CollectSections(Property(Property(paper, "body"), "sections"), null, SectionKind.Unknown);
        row.HasBody = blocks.Any(block => block.Kind != SectionKind.Abstract);
        row.BodyWordCount = blocks.Where(block => block.Kind != SectionKind.Abstract)
            .Sum(block => TextUtil.CountWords(block.Text));
        row.SectionCount = sectionIndex;
        return new ParsedArticle(row, blocks);

        void CollectSections(JsonElement sections, string? parentPath, SectionKind parentKind)
        {
            if (sections.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined) return;
            if (sections.ValueKind != JsonValueKind.Array)
                throw new InvalidDataException("Sections must be an array.");

            foreach (var section in sections.EnumerateArray())
            {
                var heading = ReadString(section, "header title");
                var kind = string.Equals(heading, "Abstract", StringComparison.OrdinalIgnoreCase)
                    ? SectionKind.Abstract : SectionClassifier.Classify(null, heading);
                if (kind == SectionKind.Unknown) kind = parentKind;
                if (SectionClassifier.IsNonEvidential(kind) ||
                    string.Equals(heading, "References", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(heading, "Bibliography", StringComparison.OrdinalIgnoreCase))
                    continue;

                var path = string.IsNullOrEmpty(parentPath) ? heading
                    : string.IsNullOrEmpty(heading) ? parentPath : parentPath + " / " + heading;
                var currentIndex = sectionIndex++;
                foreach (var text in ReadText(Property(section, "text")))
                    blocks.Add(new ParsedBlock(kind, heading, path, currentIndex, text));

                var tables = Property(section, "tables");
                if (tables.ValueKind == JsonValueKind.Array)
                {
                    foreach (var table in tables.EnumerateArray())
                    {
                        var caption = table.ValueKind == JsonValueKind.Object
                            ? ReadString(table, "caption") : null;
                        if (!string.IsNullOrWhiteSpace(caption))
                            blocks.Add(new ParsedBlock(SectionKind.TableCaption, heading, path, sectionIndex++, caption));
                    }
                }

                CollectSections(Property(section, "subsections"), path, kind);
            }
        }
    }

    private static JsonElement Property(JsonElement value, string name) =>
        value.ValueKind == JsonValueKind.Object && value.TryGetProperty(name, out var property) ? property : default;

    private static string? ReadString(JsonElement value, string name)
    {
        var property = Property(value, name);
        if (property.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined) return null;
        if (property.ValueKind != JsonValueKind.String)
            throw new InvalidDataException($"'{name}' must be a string or null.");
        var text = TextUtil.Clean(property.GetString()!);
        return text.Length == 0 ? null : text;
    }

    private static IEnumerable<string> ReadText(JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Null:
            case JsonValueKind.Undefined:
                yield break;
            case JsonValueKind.String:
                var text = TextUtil.Clean(value.GetString()!);
                if (text.Length > 0) yield return text;
                yield break;
            case JsonValueKind.Array:
                foreach (var item in value.EnumerateArray())
                    foreach (var itemText in ReadText(item)) yield return itemText;
                yield break;
            case JsonValueKind.Object when value.TryGetProperty("text", out var sentence):
                foreach (var sentenceText in ReadText(sentence)) yield return sentenceText;
                yield break;
            default:
                throw new InvalidDataException($"Expected text, sentence objects, or nested text arrays, got {value.ValueKind}.");
        }
    }
}