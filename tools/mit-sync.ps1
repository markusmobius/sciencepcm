<#
.SYNOPSIS
    Builds the MIT Press passage corpus on nerds21 and syncs it to RemoteBlobStore.

.DESCRIPTION
    The MITMCP counterpart to nerds21-sync.ps1. Same ingest, same Parquet schema, same
    uploader - only the input changes: the 10 MIT Press journals exported to
    neuroscience_archive_for_ms instead of the PMC/bioRxiv/medRxiv dataset.

    Each journal folder becomes its own corpus label, so journal_id lands in the
    SourceCorpus column.

    The archive spans 1989-2025 and no year filter is applied by default. Roughly half
    the articles are XML metadata only; those carry no passages but do carry an abstract,
    so the ingest writes two tiers - passages from the full-text half, abstracts from
    everything - mirroring how the neuroscience service pairs its deep passage index with
    a broad OpenAlex abstract index.

    Idempotent. The ingest is skipped when its report already exists, and the sync skips
    any cloud directory that already holds every local file.

.EXAMPLE
    .\tools\mit-sync.ps1 -CheckOnly
    Reports what would be built and transferred, without doing either.

.EXAMPLE
    .\tools\mit-sync.ps1
    Builds if needed, then syncs.

.EXAMPLE
    .\tools\mit-sync.ps1 -Force
    Re-runs the ingest even if output already exists, and re-uploads.
#>

[CmdletBinding()]
param(
    [string]$Root = 'D:\SciencePCM',
    [string]$Python = 'D:\SciencePCM\mcpserver\venvs\sync\Scripts\python.exe',
    [int]$YearMin = 0,
    [int]$YearMax = 0,
    [int]$Threads = 256,
    [switch]$Force,
    [switch]$CheckOnly,
    [switch]$SkipIngest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
$content = Join-Path $Root 'neuroscience_archive_for_ms\content'

# Same disposable derived-data root the neuroscience sync uses; the cloud prefix differs.
$temp = Join-Path $Root 'mcpserver\__temp'
$passages = Join-Path $temp 'mitp-passages'
$abstracts = Join-Path $temp 'mitp-abstracts'

function Write-Step($text) {
    Write-Host ''
    Write-Host "=== $text" -ForegroundColor Cyan
}

function Assert-LastExit($what) {
    if ($LASTEXITCODE -ne 0) {
        throw "$what failed with exit code $LASTEXITCODE."
    }
}

Write-Step 'Checking prerequisites'

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw 'dotnet was not found on PATH.'
}

if (-not (Test-Path $content)) {
    throw "MIT archive not found at $content. Check -Root."
}

if (-not (Test-Path $Python)) {
    Write-Host "  python venv missing at $Python" -ForegroundColor Yellow
    if ($CheckOnly) {
        Write-Host '  would create it (re-run without -CheckOnly)'
    }
    else {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "No venv at $Python and uv is not installed. Install uv, or pass -Python <path>."
        }
        $venvRoot = Split-Path -Parent (Split-Path -Parent $Python)
        Write-Host "  creating venv at $venvRoot ..."
        & uv venv $venvRoot --python 3.12
        Assert-LastExit 'uv venv'
    }
}
if ((Test-Path $Python) -and -not $env:legopds_clienthash -and -not $env:CLOUDPDS_CLIENT_HASH) {
    throw 'Set legopds_clienthash (or CLOUDPDS_CLIENT_HASH) before syncing.'
}

# One corpus label per journal directory, so journal_id survives into SourceCorpus.
$journals = Get-ChildItem $content -Directory | Sort-Object Name
if ($journals.Count -eq 0) {
    throw "No journal directories under $content."
}
foreach ($journal in $journals) {
    $count = (Get-ChildItem $journal.FullName -Filter *.xml -File).Count
    Write-Host ("  {0,-8} {1,6:N0} xml" -f $journal.Name, $count)
}

Write-Host '  RemoteBlobStore ... ' -NoNewline
if (-not (Test-Path $Python)) {
    Write-Host 'skipped (no venv yet)'
}
else {
    & $Python -c "import RemoteBlobStore" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'not installed, installing' -ForegroundColor Yellow
        if (-not $CheckOnly) {
            # uv-created venvs have no pip, so install through uv when it is available.
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                & uv pip install --python $Python "git+https://github.com/markusmobius/newsprinceton-pythoncloud"
            }
            else {
                & $Python -m pip install --quiet "git+https://github.com/markusmobius/newsprinceton-pythoncloud"
            }
            Assert-LastExit 'install RemoteBlobStore'
        }
    }
    else {
        Write-Host 'ok'
    }
}

$report = Join-Path $passages 'ingest-report.json'

Write-Step 'Ingest'

if ($SkipIngest) {
    Write-Host '  -SkipIngest set'
}
elseif ((Test-Path $report) -and -not $Force) {
    $counts = (Get-Content $report -Raw | ConvertFrom-Json).counts
    Write-Host "  already built: $($counts.articles_written) articles, $($counts.chunks_written) chunks, $($counts.abstracts_written) abstracts"
    Write-Host '  use -Force to rebuild'
}
elseif ($CheckOnly) {
    Write-Host "  would ingest $($journals.Count) journals into $passages"
}
else {
    Write-Host '  building SciencePcm.Ingest ...'
    Push-Location $repo
    try {
        & dotnet build -c Release --nologo -v q
        Assert-LastExit 'dotnet build'

        # A rebuild must not inherit shards from a wider run: the writer overwrites
        # part-NNNN in place, so a shorter run would leave stale tail shards behind.
        if ($Force) {
            foreach ($dir in @($passages, $abstracts)) {
                if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
            }
        }

        $arguments = @()
        foreach ($journal in $journals) {
            $arguments += '--input'
            $arguments += "$($journal.Name)=$($journal.FullName)"
        }
        $arguments += @('--out', $passages, '--abstracts-out', $abstracts, '--threads', $Threads)
        if ($YearMin -gt 0) { $arguments += @('--year-min', $YearMin) }
        if ($YearMax -gt 0) { $arguments += @('--year-max', $YearMax) }

        Write-Host '  running ingest ...'
        & dotnet run --project (Join-Path $repo 'src\SciencePcm.Ingest') -c Release -- @arguments
        Assert-LastExit 'ingest'
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path $report)) {
        throw "Ingest produced no report at $report. Refusing to sync."
    }
}

Write-Step 'Sync'

if (-not (Test-Path $Python)) {
    Write-Host '  skipped: no python venv (re-run without -CheckOnly to create it)'
    return
}

$syncArguments = @((Join-Path $PSScriptRoot 'cloudstore.py'), 'sync', '--corpus', 'mitp')
if ($CheckOnly) { $syncArguments += '--check' }
# A rebuilt corpus reuses the shard filenames, so the name check would skip it.
if ($Force) { $syncArguments += '--force' }

# RemoteBlobServer parses its port from the first stdout line, but prints a MAXCORES
# warning there first when the variable is set, which breaks startup. Cleared for the
# duration and restored afterwards.
$savedMaxCores = $env:MAXCORES
if ($null -ne $savedMaxCores) {
    Write-Host "  clearing MAXCORES ($savedMaxCores) for the blob server"
    Remove-Item Env:MAXCORES -ErrorAction SilentlyContinue
}

Push-Location $repo
try {
    & $Python @syncArguments
    Assert-LastExit 'cloudstore.py sync'

    if (-not $CheckOnly) {
        Write-Step 'Verifying'
        & $Python (Join-Path $PSScriptRoot 'cloudstore.py') sync --corpus mitp --check
        Assert-LastExit 'cloudstore.py sync --check'
    }
}
finally {
    Pop-Location
    if ($null -ne $savedMaxCores) { $env:MAXCORES = $savedMaxCores }
}

Write-Step 'Done'
Write-Host "  passages  : $passages -> mitmcp/passages"
Write-Host "  abstracts : $abstracts -> mitmcp/abstracts"
if (-not $CheckOnly) {
    Write-Host '  Once the verify above reports "all files present", the local copy is disposable.'
}
