using SciencePcm.Core;
using SciencePcm.Server;

namespace CustomMcp.Server;

public sealed class CustomCorpusRegistry : IDisposable
{
    private readonly List<CustomCollection> _collections = [];
    public IReadOnlyList<CustomCollection> Collections => _collections.AsReadOnly();

    public CustomCorpusRegistry(IReadOnlyList<PdsCorpus> corpora, string indexRoot, ServerOptions options, SharedReranker reranker)
    {
        try
        {
            foreach (var corpus in corpora)
            {
                var prepared = PreparedPdsCorpus.Load(corpus, indexRoot);
                var directory = prepared.IndexDirectory(indexRoot);
                var retrieval = new RetrievalService(options with
                {
                    IndexPath = Path.Combine(directory, "abstracts"),
                    PassageIndexPath = Path.Combine(directory, "passages"),
                }, reranker);
                _collections.Add(new CustomCollection(corpus, prepared, retrieval));
                if (retrieval.DocumentCount != prepared.Documents || retrieval.PassageCount != prepared.Passages)
                    throw new InvalidDataException($"Prepared index counts do not match '{corpus.Name}'; run prepare again.");
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        foreach (var collection in _collections) collection.Retrieval.Dispose();
        _collections.Clear();
    }
}

public sealed record CustomCollection(PdsCorpus Definition, PreparedPdsCorpus Prepared, RetrievalService Retrieval);