using System.Text.Json;
using System.Text.Json.Nodes;
using CustomMcp.Ingest;
using Pds;
using SciencePcm.Core;
using Xunit;

namespace MitPcm.Tests;

[CollectionDefinition("MIT corpus")]
public sealed class MitCorpusCollection : ICollectionFixture<MitCorpusFixture>;

public sealed class MitCorpusFixture : IAsyncLifetime
{
    public const string PaperKey = "papers:paper-one";
    public const string ErratumKey = "papers:erratum";
    public const string Abstract = "Neurons encode sparse patterns. Associative memory recovers noisy patterns.";
    public string Root { get; } = Path.Combine(Path.GetTempPath(), "mitpcm-tests-" + Guid.NewGuid().ToString("N"));
    public string Input => Path.Combine(Root, "source.pds");
    public string Output => Path.Combine(Root, "corpus");
    public string AbstractGlob => Path.Combine(Output, "abstracts", "abstracts-part-*.parquet");
    public string MetadataGlob => Path.Combine(Output, "passages", "articles-part-*.parquet");
    public string ChunkGlob => Path.Combine(Output, "passages", "chunks-part-*.parquet");
    public string IndexRoot => Path.Combine(Root, "indexes");
    public string Config => Path.Combine(Root, "config.json");
    public string PaperIndex => Path.Combine(IndexRoot, "papers", new string('a', 64), "abstracts");
    public string PassageIndex => Path.Combine(IndexRoot, "papers", new string('a', 64), "passages");
    public string Model => Path.Combine(Root, "test-only-model");
    public PdsIngestResult Result { get; private set; } = null!;

    public async Task InitializeAsync()
    {
        Directory.CreateDirectory(Root);
        CreateInput(Input);
        using (var reader = new PdsReader())
        {
            reader.Open(Input);
            Result = await PdsIngestor.WriteAsync(reader, Input, "test/source.pds", Output, "papers", shardSize: 1);
        }
        Assert.Equal(0, await SciencePcm.Index.Program.Main(["build", "--schema", "abstracts", "--input", AbstractGlob,
            "--metadata", MetadataGlob, "--out", PaperIndex, "--threads", "1", "--ram-buffer", "32"]));
        Assert.Equal(0, await SciencePcm.Index.Program.Main(["build", "--schema", "chunks", "--input", ChunkGlob,
            "--metadata", MetadataGlob, "--out", PassageIndex, "--threads", "1", "--ram-buffer", "32"]));

        await new PreparedPdsCorpus
        {
            Name = "papers", InputCloudPath = "test/source.pds", InputSha256 = Result.SourceSha256,
            Generation = new string('a', 64), Documents = Result.Articles, Passages = Result.Chunks,
        }.PublishAsync(IndexRoot);
        var otherInput = Path.Combine(Root, "other.pds");
        using (var writer = new PdsWriter(PdsCompressionMode.ZstdNoDict))
        {
            writer.SetMetadataJson("{}");
            using var other = JsonDocument.Parse("""
                {"metadata":{"title":"Distinct control study","authors":["Other Author"]},
                 "abstract":[{"text":"Neurons provide memory in a distinct control study."}],
                 "body":{"sections":[{"header title":"Methods","text":[{"text":"We measured neurons in the control experiment."}]}]}}
                """);
            writer.Add("paper-one", new TestPdsValue { Value = other.RootElement.Clone() });
            writer.Save(otherInput);
        }
        await PdsPreparation.PrepareAsync(new PdsCorpus { Name = "other", InputCloudPath = "test/other.pds" },
            otherInput, Path.Combine(Root, "other-data"), IndexRoot, threads: 1);
        await File.WriteAllTextAsync(Config, """
            {"customCorpora":[{"name":"papers","inputCloudPath":"test/source.pds"},{"name":"other","inputCloudPath":"test/other.pds"}]}
            """);

        Directory.CreateDirectory(Model);
        const string tokenSumModel = "CAgSDG1pdHBjbS10ZXN0czq2AgokCglpbnB1dF9pZHMSBnRva2VucyIEQ2FzdCoJCgJ0bxgBoAECCjIKBnRva2VucwoEYXhlcxIGbG9naXRzIglSZWR1Y2VTdW0qDwoIa2VlcGRpbXMYAaABAhIkdGVzdF90b2tlbl9zdW1fbm90X2FfcmVsZXZhbmNlX21vZGVsKg0IARAHOgEBQgRheGVzWigKCWlucHV0X2lkcxIbChkIBxIVCgcSBWJhdGNoCgoSCHNlcXVlbmNlWi0KDmF0dGVudGlvbl9tYXNrEhsKGQgHEhUKBxIFYmF0Y2gKChIIc2VxdWVuY2VaLQoOdG9rZW5fdHlwZV9pZHMSGwoZCAcSFQoHEgViYXRjaAoKEghzZXF1ZW5jZWIdCgZsb2dpdHMSEwoRCAESDQoHEgViYXRjaAoCCAFCBAoAEA0=";
        await File.WriteAllBytesAsync(Path.Combine(Model, "model.onnx"), Convert.FromBase64String(tokenSumModel));
        await File.WriteAllTextAsync(Path.Combine(Model, "vocab.txt"), "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nneurons\npatterns\nmemory\n.\n");
    }

    public static void CreateInput(string path, bool invalidTitle = false)
    {
        using var writer = new PdsWriter(PdsCompressionMode.ZstdNoDict);
        writer.SetMetadataJson("""
            {"journal":["Neural Computation","unknown"],"checkpoint_records":[
                {"paper_key":"paper-one","filename":"neco_a_01243.pdf"},
                {"paper_key":"erratum","filename":"neco_c_01397.pdf"}]}
            """);
        var paper = JsonNode.Parse("""
            {
              "metadata":{"title":"Sparse neural memory","authors":["Luis Sa-Couto","Andreas Wichert"],
                          "journal":null,"publish date":"2024-05-10","bibliography":["Reference not used as evidence."]},
              "abstract":[{"sentence_index":1,"text":"Neurons encode sparse patterns."},
                          {"sentence_index":2,"text":"Associative memory recovers noisy patterns."}],
              "body":{"sections":[
                {"header title":"1 Introduction","text":[[{"sentence_index":1,"text":"Neurons provide an associative memory for noisy patterns."}]],"tables":[],"subsections":[]},
                {"header title":"Methods","text":[[{"sentence_index":2,"text":"We trained neurons using a controlled plasticity protocol and measured retrieval of the stored patterns."}]],"tables":[],"subsections":[]},
                {"header title":"References","text":["This reference must not become a passage."],"tables":[],"subsections":[]}
              ]}
            }
            """)!;
        if (invalidTitle) paper["metadata"]!["title"] = null;
        writer.Add("paper-one", new TestPdsValue { Value = JsonSerializer.SerializeToElement(paper) });
        using var erratum = JsonDocument.Parse("""
            {"metadata":{"title":"Erratum to Sparse neural memory","authors":["Richard Gast","Helmut Schmidt","Thomas R. Kn\u00f6sche"],
                         "journal":"Neural Computation","publish date":null,"bibliography":[]},
             "abstract":[],"body":{"sections":[{"header title":"Erratum","text":[[{"sentence_index":1,"text":"The equation needs a corrected scale factor."}]],"tables":[],"subsections":[]}]}}
            """);
        writer.Add("erratum", new TestPdsValue { Value = erratum.RootElement.Clone() });
        writer.Save(path);
    }

    public Task DisposeAsync()
    {
        Directory.Delete(Root, recursive: true);
        return Task.CompletedTask;
    }

    private sealed class TestPdsValue : IPdsJsonValue
    {
        public JsonElement Value { get; set; }
        public void ReadJson(JsonElement element) => Value = element.Clone();
        public void WriteJson(Utf8JsonWriter writer) => Value.WriteTo(writer);
    }
}