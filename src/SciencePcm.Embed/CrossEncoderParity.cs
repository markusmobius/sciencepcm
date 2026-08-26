using System.Text.Json;

namespace SciencePcm.Embed;

/// <summary>
/// Replays the pair samples written by tools/export_onnx.py.
///
/// Both reranker families assemble pairs in C# rather than in the tokenizer, because
/// neither tokenizer backend offers a pair overload. The layouts differ - BERT uses
/// [CLS] A [SEP] B [SEP] with segment ids, XLM-R uses bos A eos eos B eos with none -
/// and getting either wrong degrades every score silently instead of failing. So the
/// exporter records what HuggingFace produced and this checks we reproduce it.
/// </summary>
public static class CrossEncoderParity
{
    public static (int Passed, int Total, List<string> Failures) Verify(string parityPath, ICrossEncoder encoder)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(parityPath));
        var root = document.RootElement;

        var failures = new List<string>();
        var passed = 0;
        var total = 0;

        foreach (var sample in root.GetProperty("samples").EnumerateArray())
        {
            if (!sample.TryGetProperty("query", out var queryElement)) continue;

            total++;
            var query = queryElement.GetString() ?? "";
            var passage = sample.GetProperty("passage").GetString() ?? "";
            var expectedIds = sample.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();

            // Only BERT-family exports record segment ids.
            var expectedTypes = sample.TryGetProperty("token_type_ids", out var types)
                ? types.EnumerateArray().Select(e => e.GetInt32()).ToArray()
                : null;

            var (ids, typeIds) = encoder.EncodePair(query, passage);

            var idsMatch = ids.SequenceEqual(expectedIds);
            var typesMatch = expectedTypes is null || typeIds.SequenceEqual(expectedTypes);

            if (idsMatch && typesMatch)
            {
                passed++;
                continue;
            }

            var report = $"\"{query}\" + \"{passage[..Math.Min(40, passage.Length)]}...\"\n" +
                         $"    expected ids : {string.Join(",", expectedIds)}\n" +
                         $"    actual ids   : {string.Join(",", ids)}";

            if (expectedTypes is not null && !typesMatch)
            {
                report += $"\n    expected types : {string.Join(",", expectedTypes)}" +
                          $"\n    actual types   : {string.Join(",", typeIds)}";
            }

            failures.Add(report);
        }

        return (passed, total, failures);
    }
}
