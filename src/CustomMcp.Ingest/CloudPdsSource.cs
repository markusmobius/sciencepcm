using AdvancedLogger;
using BlobClient;
using BlobClient.Download;
using BlobDataMachine;
using BlobMessages.downloads;
using FileFilters.directories;
using SciencePcm.Core;

namespace CustomMcp.Ingest;

public static class CloudPdsSource
{
    public static string CachedPath(string cloudPath, string cacheDirectory)
    {
        PdsCorpus.ValidateCloudPath(cloudPath);
        return Path.Combine(Path.GetFullPath(cacheDirectory), Path.Combine(cloudPath.Split('/')));
    }

    public static async Task<string> DownloadAsync(string cloudPath, string cacheDirectory, bool refresh = false)
    {
        var localPath = CachedPath(cloudPath, cacheDirectory);
        if (!refresh && File.Exists(localPath)) return localPath;
        cacheDirectory = Path.GetFullPath(cacheDirectory);

        var clientHash = Environment.GetEnvironmentVariable("legopds_clienthash");
        if (string.IsNullOrWhiteSpace(clientHash)) clientHash = Environment.GetEnvironmentVariable("CLOUDPDS_CLIENT_HASH");
        if (string.IsNullOrWhiteSpace(clientHash))
            throw new InvalidOperationException("Set legopds_clienthash before downloading the input, or use --offline with an existing cache.");

        const string serverUrl = "https://www.legopds.projectratio.net:6008";
        var segments = cloudPath.Split('/');
        var machine = new dataVersionMachine(clientHash, serverUrl, cacheDirectory);
        var directory = machine.directory(string.Join('/', segments[..^1]));
        if (!directory.files.ContainsKey(segments[^1]))
            throw new FileNotFoundException($"Cloud input not found: {cloudPath}");

        var staging = Path.Combine(cacheDirectory, ".downloads", Guid.NewGuid().ToString("N"));
        var log = new advancedLogger();
        try
        {
            Directory.CreateDirectory(staging);
            var task = new DownloadTask
            {
                localRootDirectory = staging,
                downloads = new DirectorySet { paths = [cloudPath + "|"] },
            };
            Console.WriteLine($"Downloading {cloudPath}");
            var authentication = new Authentication { clienthash = clientHash, serverURL = serverUrl };
            if (!await new DownloaderClient(authentication).download(task, log))
                throw new IOException("The cloud download failed; the previous cached file has not been changed.");
            var downloaded = CachedPath(cloudPath, staging);
            if (!File.Exists(downloaded)) throw new FileNotFoundException("The downloaded input is missing.", downloaded);
            using (var reader = new Pds.PdsReader()) reader.Open(downloaded);
            Directory.CreateDirectory(Path.GetDirectoryName(localPath)!);
            File.Move(downloaded, localPath, overwrite: true);
            return localPath;
        }
        finally
        {
            log.close();
            if (Directory.Exists(staging)) Directory.Delete(staging, recursive: true);
        }
    }
}