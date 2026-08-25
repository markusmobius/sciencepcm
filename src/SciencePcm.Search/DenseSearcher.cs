using Cloud.Unum.USearch;
using SciencePcm.Embed;

namespace SciencePcm.Search;

public sealed record Hit(string Id, string ArticleKey, float Score);

/// <summary>
/// A usearch index plus the id list written beside it. Index key N is line N+1 of the
/// keys file, which is the only thing tying a vector back to a document.
/// </summary>
public sealed class DenseSearcher : IDisposable
{
    private readonly USearchIndex _index;
    private readonly string[] _ids;
    private readonly TextEmbedder _embedder;
    private readonly int _fetchMultiplier;

    public DenseSearcher(
        string indexPath,
        string keysPath,
        TextEmbedder embedder,
        bool view = true,
        int fetchMultiplier = 4)
    {
        _index = new USearchIndex(indexPath, view);
        _ids = File.ReadAllLines(keysPath);
        _embedder = embedder;
        _fetchMultiplier = Math.Max(1, fetchMultiplier);

        if ((ulong)_ids.Length != _index.Size())
        {
            throw new InvalidOperationException(
                $"Key file has {_ids.Length:N0} lines but the index holds {_index.Size():N0} vectors. " +
                "They are not a matching pair.");
        }
    }

    public ulong Size => _index.Size();

    public ulong Dimensions => _index.Dimensions();

    public float[] Encode(string query) => _embedder.Embed(new[] { query })[0];

    /// <summary>Passage-level hits, best first.</summary>
    public List<Hit> Search(string query, int k) => Search(Encode(query), k);

    public List<Hit> Search(float[] queryVector, int k)
    {
        var count = _index.Search(queryVector, k, out var keys, out var distances);
        var hits = new List<Hit>(count);

        for (var i = 0; i < count; i++)
        {
            var id = _ids[keys[i]];
            // Cos metric stores 1 - cosine, and the vectors are L2-normalised.
            hits.Add(new Hit(id, ArticleKeyOf(id), 1f - distances[i]));
        }

        return hits;
    }

    /// <summary>
    /// Article-level hits. BioASQ judges documents, not passages, so several chunks of
    /// one paper must collapse to a single ranked entry keeping its best score.
    /// </summary>
    public List<Hit> SearchArticles(string query, int k)
    {
        var passages = Search(query, k * _fetchMultiplier);
        var best = new Dictionary<string, Hit>(passages.Count);

        foreach (var hit in passages)
        {
            if (!best.TryGetValue(hit.ArticleKey, out var existing) || hit.Score > existing.Score)
            {
                best[hit.ArticleKey] = hit;
            }
        }

        return best.Values.OrderByDescending(h => h.Score).Take(k).ToList();
    }

    private static string ArticleKeyOf(string id)
    {
        var hash = id.IndexOf('#');
        return hash < 0 ? id : id[..hash];
    }

    public void Dispose()
    {
        _index.Dispose();
        _embedder.Dispose();
    }
}
