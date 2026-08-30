namespace OpenAlex.Index;

internal static class Program
{
    private static Task<int> Main(string[] args)
    {
        Console.WriteLine("OpenAlex abstract index");
        return global::SciencePcm.Index.Program.Main(args);
    }
}