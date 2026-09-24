using Parquet;
using Parquet.Serialization;
using SciencePcm.Core;

namespace SciencePcm.Ingest;

public static class CorpusShardWriter
{
    public static async Task<int> WriteAsync(string outputDirectory, string? abstractsDirectory, int shard,
        IReadOnlyCollection<ArticleRow> articles, IReadOnlyCollection<ChunkRow> chunks,
        bool includeWithoutAbstract = false)
    {
        var options = new ParquetOptions
        {
            CompressionMethod = CompressionMethod.Zstd,
            CompressionLevel = System.IO.Compression.CompressionLevel.Optimal,
        };

        await using (var stream = File.Create(Path.Combine(outputDirectory, $"articles-part-{shard:D4}.parquet")))
            await ParquetSerializer.SerializeAsync(articles, stream, options);
        await using (var stream = File.Create(Path.Combine(outputDirectory, $"chunks-part-{shard:D4}.parquet")))
            await ParquetSerializer.SerializeAsync(chunks, stream, options);

        if (abstractsDirectory is null) return 0;
        var abstracts = articles.Where(article => includeWithoutAbstract || !string.IsNullOrWhiteSpace(article.Abstract))
            .Select(OpenAlexShapedAbstract.From).ToList();
        await using (var stream = File.Create(Path.Combine(abstractsDirectory, $"abstracts-part-{shard:D4}.parquet")))
            await ParquetSerializer.SerializeAsync(abstracts, stream, options);
        return abstracts.Count;
    }
}