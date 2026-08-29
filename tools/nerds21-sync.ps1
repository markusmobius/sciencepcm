<#
.SYNOPSIS
    Builds the derived corpus on nerds21 and syncs it to RemoteBlobStore.

.DESCRIPTION
    One entry point for the only work that has to happen on nerds21: turning the
    62.8 GB of PMC/bioRxiv/medRxiv XML into Parquet, then uploading the compact
    result. Everything downstream - embedding, indexing, serving - runs on the
    A100 box from what this produces.

    Idempotent. The ingest is skipped when its report already exists, and the
    sync skips any cloud directory that already holds every local file.

.EXAMPLE
    .\tools\nerds21-sync.ps1 -CheckOnly
    Reports what would be built and transferred, without doing either.

.EXAMPLE
    .\tools\nerds21-sync.ps1
    Builds if needed, then syncs.

.EXAMPLE
    .\tools\nerds21-sync.ps1 -Force
    Re-runs the ingest even if output already exists.
#>

[CmdletBinding()]
param(
    [string]$Root = 'D:\SciencePCM',
    [string]$Python = 'D:\SciencePCM\mcpserver\venvs\sync\Scripts\python.exe',
    [int]$YearMin = 2019,
    [int]$YearMax = 2025,
    [int]$Threads = 256,
    [switch]$Force,
    [switch]$CheckOnly,
    [switch]$SkipIngest,
    [switch]$IncludeOptional
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
$dataset = Join-Path $Root 'dataset'

# Everything derived lives under __temp: disposable, and rebuilt here when missing.
$temp = Join-Path $Root 'mcpserver\__temp'
$passages = Join-Path $temp 'passages-2019-2025'

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

# Corpus label -> full-text root. Labels land in the SourceCorpus column.
$corpora = [ordered]@{
    'pmc'          = Join-Path $dataset 'PMC-neuroscience-2019-2025\fulltext'
    'biorxiv'      = Join-Path $dataset 'biorxiv-neuroscience-2019-2025\fulltext'
    'biorxiv-supp' = Join-Path $dataset 'biorxiv-neuroscience-2019-2025-supplement\fulltext'
    'medrxiv'      = Join-Path $dataset 'medrxiv-neuroscience-2019-2025\fulltext'
}

$missing = @()
foreach ($label in $corpora.Keys) {
    if (Test-Path $corpora[$label]) {
        Write-Host ("  {0,-14} {1}" -f $label, $corpora[$label])
    }
    else {
        Write-Host ("  {0,-14} MISSING: {1}" -f $label, $corpora[$label]) -ForegroundColor Yellow
        $missing += $label
    }
}
if ($missing.Count -eq $corpora.Count) {
    throw 'No corpora found. Check -Root.'
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
$alreadyIngested = Test-Path $report

Write-Step 'Ingest'

if ($SkipIngest) {
    Write-Host '  -SkipIngest set'
}
elseif ($alreadyIngested -and -not $Force) {
    $counts = (Get-Content $report -Raw | ConvertFrom-Json).counts
    Write-Host "  already built: $($counts.articles_written) articles, $($counts.chunks_written) chunks"
    Write-Host '  use -Force to rebuild'
}
elseif ($CheckOnly) {
    Write-Host "  would ingest $($corpora.Count - $missing.Count) corpora into $passages"
}
else {
    Write-Host '  building SciencePcm.Ingest ...'
    Push-Location $repo
    try {
        & dotnet build -c Release --nologo -v q
        Assert-LastExit 'dotnet build'

        $arguments = @()
        foreach ($label in $corpora.Keys) {
            if (Test-Path $corpora[$label]) {
                $arguments += '--input'
                $arguments += "$label=$($corpora[$label])"
            }
        }
        $arguments += @(
            '--out', $passages,
            '--year-min', $YearMin,
            '--year-max', $YearMax,
            '--threads', $Threads
        )

        Write-Host "  running ingest ($YearMin-$YearMax) ..."
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

Write-Step 'Staging read-only sources into __temp'

# The uploader opens each source file read-write to fingerprint it, which the dataset
# ACL forbids. Copying into __temp gives us files we own.
$staged = [ordered]@{
    'abstracts'      = Join-Path $dataset 'OpenAlex-neuroscience-abstracts\data'
    'questions'      = Join-Path $dataset 'Questions-neuroscience'
}
if ($IncludeOptional) {
    $staged['openalex-works'] = Join-Path $dataset 'OpenAlex-neuroscience\data'
}

foreach ($name in $staged.Keys) {
    $source = $staged[$name]
    $target = Join-Path $temp $name

    if (-not (Test-Path $source)) {
        Write-Host ("  {0,-16} source missing: {1}" -f $name, $source) -ForegroundColor Yellow
        continue
    }

    $sourceCount = (Get-ChildItem $source -Recurse -File).Count
    $targetCount = if (Test-Path $target) { (Get-ChildItem $target -Recurse -File).Count } else { 0 }

    if ($targetCount -eq $sourceCount) {
        Write-Host ("  {0,-16} already staged ({1} files)" -f $name, $targetCount)
    }
    elseif ($CheckOnly) {
        Write-Host ("  {0,-16} would stage {1} files" -f $name, $sourceCount)
    }
    else {
        Write-Host ("  {0,-16} copying {1} files ..." -f $name, $sourceCount)
        New-Item -ItemType Directory -Force $target | Out-Null
        Copy-Item -Path (Join-Path $source '*') -Destination $target -Recurse -Force
        Write-Host ("  {0,-16} staged" -f $name)
    }
}

Write-Step 'Sync'

if (-not (Test-Path $Python)) {
    Write-Host '  skipped: no python venv (re-run without -CheckOnly to create it)'
    return
}

$syncArguments = @((Join-Path $PSScriptRoot 'cloudstore.py'), 'sync')
if ($CheckOnly) { $syncArguments += '--check' }
if ($IncludeOptional) { $syncArguments += '--include-optional' }
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
        & $Python (Join-Path $PSScriptRoot 'cloudstore.py') sync --check
        Assert-LastExit 'cloudstore.py sync --check'
    }
}
finally {
    Pop-Location
    if ($null -ne $savedMaxCores) { $env:MAXCORES = $savedMaxCores }
}

Write-Step 'Done'
if (-not $CheckOnly) {
    Write-Host '  Once every entry above reports "all files present", __temp is disposable:'
    Write-Host "    Remove-Item -Recurse -Force '$temp'"
    Write-Host '  Re-running this script rebuilds whatever is missing.'
    Write-Host '  Keep mcpserver\repo and mcpserver\venvs.'
}
