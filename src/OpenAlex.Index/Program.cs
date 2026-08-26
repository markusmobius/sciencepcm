namespace OpenAlex.Index;

internal static class Program
{
    private static Task<int> Main(string[] args)
    {
        Console.WriteLine("OpenAlex abstract index");
        return SciencePcm.Lexical.Program.Main(args);
    }
}