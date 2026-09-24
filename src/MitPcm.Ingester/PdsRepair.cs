using System.Globalization;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Xml;
using System.Xml.Linq;
using AdvancedLogger;
using BlobClient;
using BlobClient.Upload;
using BlobDataMachine;
using Pds;
using SciencePcm.Core;

namespace MitPcm.Ingester;

internal static class PdsRepair
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    public static async Task<int> PrepareAsync(PdsReader reader, string sourcePath, string directory, int? limit)
    {
        if (limit is <= 0) throw new ArgumentException("--limit must be positive.");
        var referencesDirectory = Path.Combine(directory, "references");
        Directory.CreateDirectory(referencesDirectory);
        var checkpoints = reader.Metadata.GetProperty("checkpoint_records").EnumerateArray()
            .ToDictionary(record => record.GetProperty("paper_key").GetString()!, StringComparer.Ordinal);
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(45) };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("SciencePCM-PdsRepair/1.0");
        var papers = new List<PaperRepair>();
        var paths = reader.GetKeyPaths().Take(limit ?? int.MaxValue).ToArray();
        if (limit is null)
        {
            var requested = paths.Select(keyPath => string.Join('/', keyPath)).Select(key =>
                "10.1162/" + Path.GetFileNameWithoutExtension(checkpoints[key].GetProperty("filename").GetString()!))
                .ToHashSet(StringComparer.OrdinalIgnoreCase);
            await CacheJournalAsync(client, directory, referencesDirectory, requested);
        }
        foreach (var keyPath in paths)
        {
            var key = string.Join('/', keyPath);
            var paper = reader.ReadKey<JsonPdsValue>(keyPath).Value;
            var metadata = paper.GetProperty("metadata");
            var title = metadata.GetProperty("title").GetString()!;
            if (!checkpoints.TryGetValue(key, out var checkpoint))
                throw new InvalidDataException($"No source checkpoint for {key}.");
            var filename = checkpoint.GetProperty("filename").GetString()!;
            if (Path.GetFileName(filename) != filename || !filename.StartsWith("neco_", StringComparison.Ordinal) ||
                !filename.EndsWith(".pdf", StringComparison.Ordinal))
                throw new InvalidDataException($"Unrecognised source filename: {filename}");

            var doi = "10.1162/" + Path.GetFileNameWithoutExtension(filename);
            var referencePath = Path.Combine(referencesDirectory, Path.GetFileNameWithoutExtension(filename) + ".json");
            var issues = new List<string>();
            string? referenceTitle = null;
            string? referenceAbstract = null;
            string? referenceType = null;
            string[] referenceAuthors = [];
            try
            {
                if (!File.Exists(referencePath))
                {
                    var response = await client.GetStringAsync("https://api.crossref.org/works/" + Uri.EscapeDataString(doi));
                    using var validation = JsonDocument.Parse(response);
                    if (!string.Equals(validation.RootElement.GetProperty("message").GetProperty("DOI").GetString(),
                            doi, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidDataException("Crossref returned a different DOI.");
                    await File.WriteAllTextAsync(referencePath, response);
                }
                using var reference = JsonDocument.Parse(await File.ReadAllTextAsync(referencePath));
                var work = reference.RootElement.GetProperty("message");
                referenceTitle = work.GetProperty("title")[0].GetString();
                referenceType = work.GetProperty("type").GetString();
                if (Normalise(title) != Normalise(referenceTitle ?? "")) issues.Add("title_mismatch");
                if (work.TryGetProperty("author", out var authors))
                    referenceAuthors = authors.EnumerateArray().Select(author =>
                        TextUtil.Clean(string.Join(" ", new[] { String(author, "given"), String(author, "family") }
                            .Where(part => !string.IsNullOrWhiteSpace(part)))))
                        .Where(author => author.Length > 0).ToArray();
                if (referenceAuthors.Length == 0) issues.Add("missing_reference_authors");
                if (work.TryGetProperty("abstract", out var abstractValue))
                    referenceAbstract = ReadAbstract(abstractValue.GetString()!);
                if (string.IsNullOrWhiteSpace(referenceAbstract)) issues.Add("missing_reference_abstract");
            }
            catch (HttpRequestException exception) when (exception.StatusCode == HttpStatusCode.TooManyRequests)
            {
                throw new InvalidOperationException("Crossref rate limit reached; cached references were preserved. Retry after the service limit resets.", exception);
            }
            catch (Exception exception) when (exception is HttpRequestException or TaskCanceledException or XmlException)
            {
                issues.Add("reference_error: " + exception.Message);
            }
            papers.Add(new PaperRepair(keyPath, filename, title,
                metadata.GetProperty("authors").EnumerateArray().Select(author => author.GetString()!).ToArray(),
                doi, referenceTitle, referenceType, referenceAuthors, referenceAbstract, issues.ToArray(),
                ReadSections(paper.GetProperty("body").GetProperty("sections"), ["body", "sections"]).ToArray()));
            if (papers.Count % 20 == 0 || papers.Count == paths.Length)
                Console.WriteLine($"References: {papers.Count}/{paths.Length}; review needed: {papers.Count(paper => paper.Issues.Length > 0)}");
        }

        using var input = File.OpenRead(sourcePath);
        var sourceHash = Convert.ToHexString(await SHA256.HashDataAsync(input)).ToLowerInvariant();
        var plan = new RepairPlan(sourcePath, sourceHash, papers.ToArray());
        var planPath = Path.Combine(directory, "repair-plan.json");
        await File.WriteAllTextAsync(planPath, JsonSerializer.Serialize(plan, JsonOptions));
        foreach (var paper in papers.Where(paper => paper.Issues.Length > 0))
            Console.WriteLine($"REVIEW {paper.SourceFilename}: {string.Join(", ", paper.Issues)}");
        Console.WriteLine($"Repair plan: {planPath}");
        Console.WriteLine("No PDS uploaded and no configuration changed.");
        return 0;
    }

    public static async Task<int> WriteAsync(PdsReader reader, string sourcePath, string directory)
    {
        var candidatePath = Path.Combine(directory, "repair-candidate.json");
        var candidate = JsonNode.Parse(await File.ReadAllTextAsync(candidatePath))!.AsObject();
        var sourceHash = await HashAsync(sourcePath);
        if (sourceHash != candidate["source_sha256"]!.GetValue<string>())
            throw new InvalidDataException("The source PDS does not match the reviewed source hash.");
        foreach (var (file, property) in new[]
                 { ("reviewed-corrections.json", "review_sha256"), ("repair-plan.json", "reference_plan_sha256") })
            if (await HashAsync(Path.Combine(directory, file)) != candidate[property]!.GetValue<string>())
                throw new InvalidDataException($"The repair input changed: {file}");

        var papers = candidate["papers"]!.AsArray().Select(value => value!.AsObject())
            .ToDictionary(paper => string.Join('/', Strings(paper["key_path"]!)), StringComparer.Ordinal);
        var paths = reader.GetKeyPaths().ToArray();
        if (!paths.Select(path => string.Join('/', path)).ToHashSet(StringComparer.Ordinal).SetEquals(papers.Keys))
            throw new InvalidDataException("The repair must contain exactly the original PDS keys.");
        var outputPath = Path.Combine(directory, "input_2026_09_23_fixed.pds");
        if (File.Exists(outputPath)) throw new IOException($"Refusing to overwrite an existing file: {outputPath}");
        var temporaryPath = outputPath + "." + Guid.NewGuid().ToString("N") + ".partial";
        var candidateHash = await HashAsync(candidatePath);
        var abstractCount = papers.Values.Count(paper => paper["abstract"]!.AsArray().Count > 0);
        var sentenceCount = papers.Values.Sum(paper => paper["abstract"]!.AsArray().Count);
        var metadata = JsonNode.Parse(reader.Metadata.GetRawText())!.AsObject();
        if (metadata.ContainsKey("repair")) throw new InvalidDataException("The input is already marked as repaired.");
        metadata["repair"] = JsonSerializer.SerializeToNode(new
        {
            format = "mit-abstract-authors-v1",
            source_sha256 = sourceHash,
            candidate_sha256 = candidateHash,
            review_sha256 = candidate["review_sha256"]!.GetValue<string>(),
            reference_plan_sha256 = candidate["reference_plan_sha256"]!.GetValue<string>(),
            reference_provider = candidate["reference_provider"]!.GetValue<string>(),
            sentence_segmenter = candidate["sentence_segmenter"]!.GetValue<string>(),
            created_at = DateTimeOffset.UtcNow,
            record_count = papers.Count,
            published_abstract_count = abstractCount,
            abstract_sentence_count = sentenceCount,
            no_published_abstract = papers.Values.Where(paper => paper["abstract"]!.AsArray().Count == 0)
                .Select(paper => new { doi = paper["reference_doi"]!.GetValue<string>(), reason = paper["abstract_status"]!.GetValue<string>() }),
        });

        var authorsChanged = 0;
        var bodyEdits = 0;
        var chunks = 0;
        try
        {
            using (var writer = new PdsWriter(reader.CompressionMode))
            {
                writer.SetMetadataJson(metadata.ToJsonString());
                foreach (var path in paths)
                {
                    var key = string.Join('/', path);
                    var original = JsonNode.Parse(reader.ReadKey<JsonPdsValue>(path).Value.GetRawText())!.AsObject();
                    var correction = papers[key];
                    if (!JsonNode.DeepEquals(original["metadata"]!["authors"], correction["authors"])) authorsChanged++;
                    bodyEdits += correction["body_abstract"]!["edits"]!.AsArray().Count;
                    var corrected = Correct(original, correction);
                    ValidateRecord(original, corrected, correction);
                    var value = new JsonPdsValue();
                    value.ReadJson(JsonSerializer.SerializeToElement(corrected));
                    writer.Add(path, value);
                }
                if (writer.Count != paths.Length) throw new InvalidDataException("PDS writer lost a record.");
                writer.Save(temporaryPath);
            }
            using (var written = new PdsReader())
            {
                written.Open(temporaryPath);
                if (written.CompressionMode != reader.CompressionMode ||
                    !written.GetKeyPaths().Select(path => string.Join('/', path)).ToHashSet(StringComparer.Ordinal).SetEquals(papers.Keys))
                    throw new InvalidDataException("The written PDS keys or compression changed.");
                if (!JsonNode.DeepEquals(metadata, JsonNode.Parse(written.Metadata.GetRawText())))
                    throw new InvalidDataException("The written store metadata did not round-trip.");
                foreach (var path in paths)
                {
                    var key = string.Join('/', path);
                    var original = JsonNode.Parse(reader.ReadKey<JsonPdsValue>(path).Value.GetRawText())!.AsObject();
                    var value = written.ReadKey<JsonPdsValue>(path).Value;
                    var actual = JsonNode.Parse(value.GetRawText())!.AsObject();
                    if (!JsonNode.DeepEquals(Correct(original, papers[key]), actual))
                        throw new InvalidDataException($"PDS record did not round-trip: {key}");
                    ValidateRecord(original, actual, papers[key]);
                    var parsed = CustomMcp.Ingest.PdsPaperParser.Parse(value, key, outputPath + "#" + key, "mit");
                    var expectedAbstract = TextUtil.Clean(string.Join(" ", papers[key]["abstract"]!.AsArray()
                        .Select(sentence => sentence!["text"]!.GetValue<string>())));
                    if ((parsed.Article.Abstract ?? "") != expectedAbstract)
                        throw new InvalidDataException($"Ingester abstract mismatch: {key}");
                    chunks += Chunker.Chunk(parsed).Count;
                }
            }
            if (await HashAsync(sourcePath) != sourceHash) throw new InvalidDataException("The original source changed during repair.");
            var outputHash = await HashAsync(temporaryPath);
            File.Move(temporaryPath, outputPath, overwrite: false);
            var report = new
            {
                output_path = outputPath,
                output_sha256 = outputHash,
                output_bytes = new FileInfo(outputPath).Length,
                source_sha256 = sourceHash,
                candidate_sha256 = candidateHash,
                records = papers.Count,
                authors_changed = authorsChanged,
                published_abstracts = abstractCount,
                abstract_sentences = sentenceCount,
                body_sentence_edits = bodyEdits,
                ingester_chunks = chunks,
                validation = new[] { "all_keys_preserved", "all_records_round_tripped", "all_author_and_abstract_fields_verified",
                    "unrelated_fields_and_body_text_preserved", "no_complete_abstract_left_in_body", "ingester_compatible", "original_sha256_unchanged" },
                uploaded = false,
                config_changed = false,
            };
            await File.WriteAllTextAsync(Path.Combine(directory, "repair-report.json"), JsonSerializer.Serialize(report, JsonOptions));
            Console.WriteLine($"Corrected PDS: {outputPath}");
            Console.WriteLine($"Verified {papers.Count} records; {abstractCount} abstracts; {sentenceCount} sentences; {authorsChanged} author lists corrected.");
            Console.WriteLine($"Body edits: {bodyEdits}; ingester chunks: {chunks}; SHA256: {outputHash}");
            Console.WriteLine("Original preserved. No PDS uploaded and no configuration changed.");
            return 0;
        }
        finally
        {
            if (File.Exists(temporaryPath)) File.Delete(temporaryPath);
        }
    }

    public static async Task<int> UploadAsync(string directory)
    {
        const string filename = "input_2026_09_23_fixed.pds";
        const string cloudDirectory = "sciencepcm/Sept222026";
        const string cloudPath = cloudDirectory + "/" + filename;
        const string serverUrl = "https://www.legopds.projectratio.net:6008";
        var reportPath = Path.Combine(directory, "repair-report.json");
        var report = JsonNode.Parse(await File.ReadAllTextAsync(reportPath))!.AsObject();
        var inputPath = Path.GetFullPath(Path.Combine(directory, filename));
        var expectedHash = report["output_sha256"]!.GetValue<string>();
        var expectedBytes = report["output_bytes"]!.GetValue<long>();
        if (!string.Equals(inputPath, Path.GetFullPath(report["output_path"]!.GetValue<string>()), StringComparison.OrdinalIgnoreCase) ||
            new FileInfo(inputPath).Length != expectedBytes || await HashAsync(inputPath) != expectedHash ||
            await HashAsync(Path.Combine(directory, "repair-candidate.json")) != report["candidate_sha256"]!.GetValue<string>())
            throw new InvalidDataException("The PDS or candidate differs from the verified repair report.");

        var clientHash = Environment.GetEnvironmentVariable("legopds_clienthash");
        if (string.IsNullOrWhiteSpace(clientHash))
            throw new InvalidOperationException("Set legopds_clienthash before uploading the repaired PDS.");
        var operationRoot = Path.GetFullPath(Path.Combine(directory, "cloud-verification", Guid.NewGuid().ToString("N")));
        var machine = new dataVersionMachine(clientHash, serverUrl, Path.Combine(operationRoot, "cache"));
        var before = machine.directory(cloudDirectory).files
            .ToDictionary(entry => entry.Key, entry => entry.Value.fileLength, StringComparer.Ordinal);
        var alreadyExists = before.ContainsKey(filename);
        if (alreadyExists)
            Console.WriteLine($"Remote file already exists; verifying without overwriting: {cloudPath}");
        else
        {
            var staging = Path.Combine(operationRoot, "upload");
            Directory.CreateDirectory(staging);
            try
            {
                var stagedPath = Path.Combine(staging, filename);
                File.Copy(inputPath, stagedPath, overwrite: false);
                if (await HashAsync(stagedPath) != expectedHash)
                    throw new InvalidDataException("The staged PDS differs from the verified artifact.");
                var log = new advancedLogger();
                try
                {
                    Console.WriteLine($"Uploading {cloudPath} ({expectedBytes:N0} bytes)");
                    var authentication = new Authentication { clienthash = clientHash, serverURL = serverUrl };
                    if (!await new UploaderClient(authentication).upload(staging, cloudDirectory, [], [], [], log,
                            with_exact: false, recursive_upload: false))
                        throw new IOException("The corrected PDS upload failed.");
                }
                finally
                {
                    log.close();
                }
            }
            finally
            {
                Directory.Delete(staging, recursive: true);
            }
        }

        var after = machine.directory(cloudDirectory).files;
        if (!after.TryGetValue(filename, out var uploaded) || uploaded.fileLength != expectedBytes)
            throw new InvalidDataException("The remote listing does not contain the expected PDS size.");
        foreach (var existing in before)
            if (!after.TryGetValue(existing.Key, out var preserved) || preserved.fileLength != existing.Value)
                throw new InvalidDataException($"An existing remote file changed: {existing.Key}");

        var downloadedPath = await Program.DownloadAsync(cloudPath, Path.Combine(operationRoot, "download"));
        var downloadedHash = await HashAsync(downloadedPath);
        if (new FileInfo(downloadedPath).Length != expectedBytes || downloadedHash != expectedHash)
            throw new InvalidDataException("The freshly downloaded cloud PDS does not match the approved artifact.");
        if (await HashAsync(inputPath) != expectedHash)
            throw new InvalidDataException("The local corrected PDS changed during the upload.");
        report["uploaded"] = true;
        report["cloud_path"] = cloudPath;
        report["remote_verified_sha256"] = downloadedHash;
        report["remote_verified_bytes"] = expectedBytes;
        report["remote_download_path"] = downloadedPath;
        report["upload_verified_at"] = DateTimeOffset.UtcNow;
        report["reused_existing_remote_file"] = alreadyExists;
        report["previous_remote_files_preserved"] = before.Count;
        await File.WriteAllTextAsync(reportPath, report.ToJsonString(JsonOptions));
        Console.WriteLine($"Cloud copy verified: {cloudPath}");
        Console.WriteLine($"Bytes: {expectedBytes:N0}; SHA256: {downloadedHash}");
        Console.WriteLine($"Preserved {before.Count} existing remote files. Configuration unchanged.");
        return 0;
    }

    private static JsonObject Correct(JsonObject original, JsonObject correction)
    {
        if (original["metadata"]!["title"]!.GetValue<string>() != correction["title"]!.GetValue<string>() ||
            !JsonNode.DeepEquals(original["metadata"]!["authors"], correction["original_authors"]))
            throw new InvalidDataException("The repair's original title or authors no longer match.");
        if (original["abstract"]!.AsArray().Count != 0) throw new InvalidDataException("Expected an unrepaired empty abstract.");
        if (Strings(correction["authors"]!).Length == 0 || Strings(correction["authors"]!).Any(string.IsNullOrWhiteSpace))
            throw new InvalidDataException("The corrected author list is empty or invalid.");
        var sentences = correction["abstract"]!.AsArray();
        for (var index = 0; index < sentences.Count; index++)
        {
            var sentence = sentences[index]!;
            var text = sentence["text"]!.GetValue<string>();
            if (sentence["sentence_index"]!.GetValue<int>() != index + 1 || string.IsNullOrWhiteSpace(text) ||
                text.Contains("[Formula: see text]", StringComparison.Ordinal))
                throw new InvalidDataException("Invalid abstract sentence or unresolved formula.");
        }
        if (sentences.Count == 0 && (correction["reference_doi"]!.GetValue<string>() != "10.1162/neco_c_01397" ||
                                    correction["abstract_status"]!.GetValue<string>() != "no_published_abstract_erratum"))
            throw new InvalidDataException("Only the reviewed erratum can have an empty abstract.");
        var corrected = original.DeepClone().AsObject();
        corrected["metadata"]!["authors"] = correction["authors"]!.DeepClone();
        corrected["abstract"] = sentences.DeepClone();
        var pending = new List<(JsonNode Node, string? Replacement)>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var edit in correction["body_abstract"]!["edits"]!.AsArray())
        {
            var path = Strings(edit!["path"]!);
            if (path.Length < 4 || path[0] != "body" || path[1] != "sections" || !path.Contains("text") ||
                !seen.Add(string.Join('/', path)))
                throw new InvalidDataException("Invalid or duplicate body edit path.");
            JsonNode node = corrected;
            foreach (var segment in path)
                node = node is JsonArray array ? array[int.Parse(segment, CultureInfo.InvariantCulture)]! : node[segment]!;
            if (node.GetValue<string>() != edit["original_text"]!.GetValue<string>())
                throw new InvalidDataException("A body edit does not match the original text.");
            pending.Add((node, edit["replacement_text"]?.GetValue<string>()));
        }
        foreach (var (node, replacement) in pending)
        {
            if (replacement is not null) node.ReplaceWith(replacement);
            else
            {
                var removed = node.Parent is JsonObject sentence && sentence.ContainsKey("sentence_index") ? sentence : node;
                if (removed.Parent is not JsonArray array || !array.Remove(removed))
                    throw new InvalidDataException("Cannot remove a reviewed abstract sentence safely.");
            }
        }
        return corrected;
    }

    private static void ValidateRecord(JsonObject original, JsonObject corrected, JsonObject correction)
    {
        var expected = original.DeepClone().AsObject();
        expected["metadata"]!["authors"] = correction["authors"]!.DeepClone();
        expected["abstract"] = correction["abstract"]!.DeepClone();
        expected["body"] = corrected["body"]!.DeepClone();
        if (!JsonNode.DeepEquals(expected, corrected) ||
            !JsonNode.DeepEquals(WithoutBodyText(original["body"]!), WithoutBodyText(corrected["body"]!)))
            throw new InvalidDataException("A field outside the reviewed repair changed.");
        using var before = JsonDocument.Parse(original.ToJsonString());
        using var after = JsonDocument.Parse(corrected.ToJsonString());
        var edits = correction["body_abstract"]!["edits"]!.AsArray()
            .ToDictionary(edit => string.Join('/', Strings(edit!["path"]!)), edit => edit!["replacement_text"]?.GetValue<string>(), StringComparer.Ordinal);
        var expectedText = ReadSections(before.RootElement.GetProperty("body").GetProperty("sections"), ["body", "sections"])
            .Select(sentence => edits.TryGetValue(string.Join('/', sentence.Path), out var replacement) ? replacement : sentence.Text)
            .Where(text => text is not null).ToArray();
        var actualText = ReadSections(after.RootElement.GetProperty("body").GetProperty("sections"), ["body", "sections"])
            .Select(sentence => sentence.Text).ToArray();
        if (!expectedText.SequenceEqual(actualText, StringComparer.Ordinal))
            throw new InvalidDataException("Retained body text or ordering changed.");
        var abstractText = Normalise(string.Join(" ", correction["abstract"]!.AsArray().Select(sentence => sentence!["text"]!.GetValue<string>())));
        if (abstractText.Length > 0 && Normalise(string.Join(" ", actualText)).Contains(abstractText, StringComparison.Ordinal))
            throw new InvalidDataException("A complete duplicate abstract remains in the body.");
    }

    private static JsonNode WithoutBodyText(JsonNode body)
    {
        var copy = body.DeepClone();
        void RemoveText(JsonArray sections)
        {
            foreach (var section in sections)
            {
                section!.AsObject().Remove("text");
                if (section["subsections"] is JsonArray children) RemoveText(children);
            }
        }
        RemoveText(copy["sections"]!.AsArray());
        return copy;
    }

    private static string[] Strings(JsonNode values) => values.AsArray().Select(value => value!.GetValue<string>()).ToArray();

    private static async Task<string> HashAsync(string path)
    {
        using var input = File.OpenRead(path);
        return Convert.ToHexString(await SHA256.HashDataAsync(input)).ToLowerInvariant();
    }

    private static IEnumerable<SourceSentence> ReadSections(JsonElement sections, string[] path)
    {
        var sectionIndex = 0;
        foreach (var section in sections.EnumerateArray())
        {
            string[] sectionPath = [.. path, (sectionIndex++).ToString(CultureInfo.InvariantCulture)];
            var heading = String(section, "header title") ?? "";
            if (section.TryGetProperty("text", out var text))
                foreach (var sentence in ReadSentences(text, [.. sectionPath, "text"], heading))
                    yield return sentence;
            if (section.TryGetProperty("subsections", out var subsections))
                foreach (var sentence in ReadSections(subsections, [.. sectionPath, "subsections"]))
                    yield return sentence;
        }
    }

    private static IEnumerable<SourceSentence> ReadSentences(JsonElement text, string[] path, string heading)
    {
        if (text.ValueKind == JsonValueKind.String)
            yield return new SourceSentence(path, heading, text.GetString()!);
        else if (text.ValueKind == JsonValueKind.Array)
        {
            var index = 0;
            foreach (var item in text.EnumerateArray())
                foreach (var sentence in ReadSentences(item, [.. path, (index++).ToString(CultureInfo.InvariantCulture)], heading))
                    yield return sentence;
        }
        else if (text.ValueKind == JsonValueKind.Object && text.TryGetProperty("text", out var sentenceText))
        {
            foreach (var sentence in ReadSentences(sentenceText, [.. path, "text"], heading))
                yield return sentence;
        }
        else if (text.ValueKind != JsonValueKind.Null)
            throw new InvalidDataException($"Unsupported body text at {string.Join('/', path)}.");
    }

    private static async Task CacheJournalAsync(HttpClient client, string directory, string referencesDirectory,
        HashSet<string> requested)
    {
        var cachePath = Path.Combine(directory, "crossref-journal.json");
        if (!File.Exists(cachePath))
        {
            Console.WriteLine("Downloading journal metadata in one batch.");
            const string url = "https://api.crossref.org/journals/0899-7667/works?filter=from-pub-date:2018-01-01&rows=1000&select=DOI,title,author,abstract,type";
            var response = await client.GetStringAsync(url);
            using var validation = JsonDocument.Parse(response);
            _ = validation.RootElement.GetProperty("message").GetProperty("items").GetArrayLength();
            await File.WriteAllTextAsync(cachePath, response);
        }
        using var batch = JsonDocument.Parse(await File.ReadAllTextAsync(cachePath));
        var matches = 0;
        foreach (var work in batch.RootElement.GetProperty("message").GetProperty("items").EnumerateArray())
        {
            var doi = work.GetProperty("DOI").GetString()!;
            if (!requested.Contains(doi)) continue;
            var filename = doi[doi.IndexOf('/')..].TrimStart('/') + ".json";
            var path = Path.Combine(referencesDirectory, filename);
            if (!File.Exists(path))
                await File.WriteAllTextAsync(path, JsonSerializer.Serialize(new { message = work }));
            matches++;
        }
        Console.WriteLine($"Journal batch matches: {matches}/{requested.Count}");
    }

    private static string? String(JsonElement value, string name) =>
        value.TryGetProperty(name, out var property) ? property.GetString() : null;

    private static string ReadAbstract(string fragment)
    {
        using var text = new StringReader("<root xmlns:jats=\"http://www.ncbi.nlm.nih.gov/JATS1\" xmlns:mml=\"http://www.w3.org/1998/Math/MathML\">" + fragment + "</root>");
        using var reader = XmlReader.Create(text, new XmlReaderSettings { DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null });
        var root = XElement.Load(reader);
        var paragraphs = root.Descendants().Where(element => element.Name.LocalName == "p").ToArray();
        return TextUtil.Clean(WebUtility.HtmlDecode(paragraphs.Length > 0
            ? string.Join(" ", paragraphs.Select(paragraph => paragraph.Value)) : root.Value));
    }

    private static string Normalise(string text)
    {
        var output = new StringBuilder();
        foreach (var character in WebUtility.HtmlDecode(text).Normalize(NormalizationForm.FormD))
        {
            if (CharUnicodeInfo.GetUnicodeCategory(character) != UnicodeCategory.NonSpacingMark && char.IsLetterOrDigit(character))
                output.Append(char.ToLowerInvariant(character));
        }
        return output.ToString();
    }

    internal sealed record RepairPlan(string SourcePath, string SourceSha256, PaperRepair[] Papers);
    internal sealed record PaperRepair(string[] KeyPath, string SourceFilename, string Title, string[] OriginalAuthors,
        string ReferenceDoi, string? ReferenceTitle, string? ReferenceType, string[] ReferenceAuthors,
        string? ReferenceAbstract, string[] Issues, SourceSentence[] SourceSentences);
    internal sealed record SourceSentence(string[] Path, string Heading, string Text);
}