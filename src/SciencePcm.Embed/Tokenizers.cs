using Microsoft.ML.Tokenizers;

namespace SciencePcm.Embed;

public enum TokenizerKind
{
    /// <summary>Microsoft.ML.Tokenizers WordPiece, driven by vocab.txt.</summary>
    MlNet,

    /// <summary>FastBertTokenizer, driven by HuggingFace tokenizer.json when present.</summary>
    Fast,
}

public interface ICorpusTokenizer
{
    /// <summary>Token ids including special tokens, truncated to the configured limit.</summary>
    int[] Encode(string text);
}

public static class TokenizerFactory
{
    public static ICorpusTokenizer Create(string modelDirectory, TokenizerKind kind, int maxTokens, bool lowerCase)
        => kind == TokenizerKind.Fast
            ? new FastTokenizer(modelDirectory, maxTokens, lowerCase)
            : new MlNetTokenizer(modelDirectory, maxTokens, lowerCase);

    public static string Describe(string modelDirectory, TokenizerKind kind)
    {
        if (kind != TokenizerKind.Fast) return "MlNet(vocab.txt)";
        return File.Exists(Path.Combine(modelDirectory, "tokenizer.json"))
            ? "Fast(tokenizer.json)"
            : "Fast(vocab.txt)";
    }
}

internal sealed class MlNetTokenizer : ICorpusTokenizer
{
    private readonly BertTokenizer _tokenizer;
    private readonly int _maxTokens;

    public MlNetTokenizer(string modelDirectory, int maxTokens, bool lowerCase)
    {
        var vocabPath = Path.Combine(modelDirectory, "vocab.txt");
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"vocab.txt not found in {modelDirectory}.", vocabPath);
        }

        _tokenizer = BertTokenizer.Create(vocabPath, new BertOptions { LowerCaseBeforeTokenization = lowerCase });
        _maxTokens = maxTokens;
    }

    public int[] Encode(string text) => _tokenizer.EncodeToIds(
        text ?? "",
        _maxTokens,
        addSpecialTokens: true,
        out _,
        out _,
        considerPreTokenization: true,
        considerNormalization: true).ToArray();
}

internal sealed class FastTokenizer : ICorpusTokenizer
{
    private readonly FastBertTokenizer.BertTokenizer _tokenizer = new();
    private readonly int _maxTokens;

    public FastTokenizer(string modelDirectory, int maxTokens, bool lowerCase)
    {
        _maxTokens = maxTokens;

        // tokenizer.json carries HuggingFace's own normalizer and pre-tokenizer settings,
        // which is what vocab.txt alone cannot express.
        var tokenizerJson = Path.Combine(modelDirectory, "tokenizer.json");
        if (File.Exists(tokenizerJson))
        {
            using var stream = File.OpenRead(tokenizerJson);
            _tokenizer.LoadTokenizerJson(stream);
            return;
        }

        var vocabPath = Path.Combine(modelDirectory, "vocab.txt");
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"Neither tokenizer.json nor vocab.txt found in {modelDirectory}.");
        }

        using var reader = new StreamReader(vocabPath);
        _tokenizer.LoadVocabulary(
            reader,
            convertInputToLowercase: lowerCase,
            unknownToken: "[UNK]",
            clsToken: "[CLS]",
            sepToken: "[SEP]",
            padToken: "[PAD]",
            normalization: System.Text.NormalizationForm.FormD);
    }

    public int[] Encode(string text)
    {
        var (ids, _, _) = _tokenizer.Encode(text ?? "", _maxTokens, null);
        var span = ids.Span;
        var result = new int[span.Length];
        for (var i = 0; i < span.Length; i++) result[i] = (int)span[i];
        return result;
    }
}
