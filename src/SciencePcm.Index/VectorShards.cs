using System.Buffers.Binary;

namespace SciencePcm.Index;

public sealed record VectorRecord(string Id, float[] Vector);

/// <summary>
/// Reads the raw shard pairs written by SciencePcm.Embed: vectors-part-NNNN.f32 holds
/// little-endian float32 rows with no header, and ids-part-NNNN.txt holds one id per
/// row in the same order.
/// </summary>
public static class VectorShards
{
    public static IReadOnlyList<(string Vectors, string Ids)> Pairs(string directory)
    {
        var pairs = new List<(string, string)>();
        foreach (var vectors in Directory.EnumerateFiles(directory, "vectors-part-*.f32")
                     .OrderBy(p => p, StringComparer.Ordinal))
        {
            var ids = Path.Combine(
                directory,
                Path.GetFileName(vectors).Replace("vectors-part-", "ids-part-").Replace(".f32", ".txt"));

            if (!File.Exists(ids))
            {
                throw new FileNotFoundException($"No id file for {Path.GetFileName(vectors)}", ids);
            }
            pairs.Add((vectors, ids));
        }

        if (pairs.Count == 0)
        {
            throw new FileNotFoundException($"No vectors-part-*.f32 found in {directory}");
        }
        return pairs;
    }

    public static int InferDimensions(string vectorsPath, string idsPath)
    {
        var rows = File.ReadLines(idsPath).Count(line => line.Length > 0);
        if (rows == 0) throw new InvalidDataException($"{idsPath} is empty.");

        var bytes = new FileInfo(vectorsPath).Length;
        if (bytes % (rows * 4L) != 0)
        {
            throw new InvalidDataException(
                $"{Path.GetFileName(vectorsPath)} is {bytes} bytes for {rows} ids, which is not a whole number of float32 rows.");
        }
        return (int)(bytes / (rows * 4L));
    }

    public static IEnumerable<VectorRecord> Read(string directory, int dimensions)
    {
        var rowBytes = dimensions * 4;
        var buffer = new byte[rowBytes];

        foreach (var (vectorsPath, idsPath) in Pairs(directory))
        {
            using var vectors = new FileStream(
                vectorsPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);

            foreach (var id in File.ReadLines(idsPath))
            {
                if (id.Length == 0) continue;

                var read = 0;
                while (read < rowBytes)
                {
                    var got = vectors.Read(buffer, read, rowBytes - read);
                    if (got == 0)
                    {
                        throw new InvalidDataException(
                            $"{Path.GetFileName(vectorsPath)} ran out of data while reading id {id}.");
                    }
                    read += got;
                }

                var vector = new float[dimensions];
                for (var i = 0; i < dimensions; i++)
                {
                    vector[i] = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(i * 4, 4));
                }

                yield return new VectorRecord(id, vector);
            }

            // A trailing remainder means the two files disagree, which would silently
            // misalign every id after this shard.
            if (vectors.Position != vectors.Length)
            {
                throw new InvalidDataException(
                    $"{Path.GetFileName(vectorsPath)} has {vectors.Length - vectors.Position} bytes "
                    + "beyond its id count. Truncate it to the id count and re-run --resume.");
            }
        }
    }
}
