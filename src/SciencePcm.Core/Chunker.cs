using System.Text;
using System.Text.RegularExpressions;

namespace SciencePcm.Core;

public sealed record ChunkOptions(int TargetWords = 300, int OverlapWords = 50, int MinWords = 30)
{
    public static ChunkOptions Default { get; } = new();
}

public static partial class Chunker
{
    [GeneratedRegex(@"(?<=[.!?])\s+(?=[""'\(\[]?[A-Z0-9])")]
    private static partial Regex SentenceBreak();

    // Tokens that end in a period but do not end a sentence.
    private static readonly HashSet<string> Abbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        "e.g", "i.e", "al", "vs", "cf", "fig", "figs", "ref", "refs", "eq", "eqs",
        "approx", "ca", "no", "nos", "sec", "min", "max", "resp", "etc", "dr", "prof",
        "s.d", "s.e.m", "ph.d", "st", "mr", "ms", "vol", "pp", "ed", "eds",
    };

    /// <summary>
    /// Packs parsed blocks into overlapping passages. Chunks never span a section
    /// boundary, so every passage keeps an unambiguous IMRaD label.
    /// </summary>
    public static List<ChunkRow> Chunk(ParsedArticle parsed, ChunkOptions? options = null)
    {
        var opts = options ?? ChunkOptions.Default;
        var results = new List<ChunkRow>();
        var chunkIndex = 0;

        foreach (var group in parsed.Blocks.GroupBy(b => b.SectionIndex))
        {
            var blocks = group.ToList();
            if (blocks.Count == 0) continue;

            var first = blocks[0];
            var combined = string.Join(" ", blocks.Select(b => b.Text));
            if (string.IsNullOrWhiteSpace(combined)) continue;

            foreach (var text in Pack(SplitSentences(combined), opts))
            {
                results.Add(new ChunkRow
                {
                    ChunkId = $"{parsed.Article.ArticleKey}#{chunkIndex:D4}",
                    ArticleKey = parsed.Article.ArticleKey,
                    SourceCorpus = parsed.Article.SourceCorpus,
                    SectionKind = first.Kind.ToString(),
                    SectionTitle = first.SectionTitle,
                    SectionPath = first.SectionPath,
                    SectionIndex = first.SectionIndex,
                    ChunkIndex = chunkIndex,
                    Text = text,
                    WordCount = TextUtil.CountWords(text),
                    Title = parsed.Article.Title,
                    PubYear = parsed.Article.PubYear,
                    IsRetracted = parsed.Article.IsRetracted,
                });
                chunkIndex++;
            }
        }

        parsed.Article.ChunkCount = results.Count;
        return results;
    }

    private static List<string> SplitSentences(string text)
    {
        var parts = SentenceBreak().Split(text);
        var merged = new List<string>(parts.Length);

        foreach (var part in parts)
        {
            if (merged.Count > 0 && EndsWithAbbreviation(merged[^1]))
            {
                merged[^1] = merged[^1] + " " + part;
            }
            else
            {
                merged.Add(part);
            }
        }

        return merged;
    }

    private static bool EndsWithAbbreviation(string sentence)
    {
        var trimmed = sentence.TrimEnd();
        if (!trimmed.EndsWith('.')) return false;

        var lastSpace = trimmed.LastIndexOf(' ');
        var token = trimmed[(lastSpace + 1)..].TrimEnd('.');
        return Abbreviations.Contains(token);
    }

    private static IEnumerable<string> Pack(List<string> sentences, ChunkOptions opts)
    {
        var chunks = new List<string>();
        var current = new List<string>();
        var currentWords = 0;

        foreach (var sentence in sentences)
        {
            var words = TextUtil.CountWords(sentence);

            if (words > opts.TargetWords * 3 / 2)
            {
                if (currentWords > 0)
                {
                    chunks.Add(string.Join(" ", current));
                    current.Clear();
                    currentWords = 0;
                }
                chunks.AddRange(HardSplit(sentence, opts));
                continue;
            }

            if (currentWords + words > opts.TargetWords && currentWords >= opts.MinWords)
            {
                chunks.Add(string.Join(" ", current));
                current = TakeOverlap(current, opts.OverlapWords);
                currentWords = current.Sum(s => TextUtil.CountWords(s));
            }

            current.Add(sentence);
            currentWords += words;
        }

        if (currentWords > 0)
        {
            // A short tail is folded into the previous chunk rather than indexed alone.
            if (currentWords < opts.MinWords && chunks.Count > 0)
            {
                chunks[^1] = chunks[^1] + " " + string.Join(" ", current);
            }
            else
            {
                chunks.Add(string.Join(" ", current));
            }
        }

        return chunks;
    }

    private static List<string> TakeOverlap(List<string> sentences, int overlapWords)
    {
        if (overlapWords <= 0) return [];

        var tail = new List<string>();
        var total = 0;

        for (var i = sentences.Count - 1; i >= 0; i--)
        {
            var words = TextUtil.CountWords(sentences[i]);
            if (total + words > overlapWords && tail.Count > 0) break;
            tail.Insert(0, sentences[i]);
            total += words;
            if (total >= overlapWords) break;
        }

        return tail;
    }

    private static IEnumerable<string> HardSplit(string sentence, ChunkOptions opts)
    {
        var words = sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var step = Math.Max(1, opts.TargetWords - opts.OverlapWords);

        for (var start = 0; start < words.Length; start += step)
        {
            var length = Math.Min(opts.TargetWords, words.Length - start);
            if (length < opts.MinWords && start > 0) break;

            var builder = new StringBuilder();
            for (var i = start; i < start + length; i++)
            {
                if (builder.Length > 0) builder.Append(' ');
                builder.Append(words[i]);
            }
            yield return builder.ToString();
        }
    }
}
