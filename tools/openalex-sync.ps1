<#
.SYNOPSIS
    Builds the all-fields OpenAlex abstract digest on nerds21 and syncs it to the cloud.

.EXAMPLE
    .\tools\openalex-sync.ps1 -CheckOnly

.EXAMPLE
    .\tools\openalex-sync.ps1

.EXAMPLE
    .\tools\openalex-sync.ps1 -Force
#>

[CmdletBinding()]
param(
    [string]$Root = 'D:\OpenAlexData',
    [string]$Python = 'D:\OpenAlexData\venvs\sync\Scripts\python.exe',
    [int]$Threads = 64,
    [int]$ShardSize = 250000,
    [switch]$Force,
    [switch]$CheckOnly,
    [switch]$SkipIngest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repo = Split-Path -Parent $PSScriptRoot
$snapshot = Join-Path $Root 'data\works'
$digest = Join-Path $Root '__temp\abstracts'
$report = Join-Path $digest 'openalex-ingest-report.json'

function Write-Step($text) {
    Write-Host ''
    Write-Host "=== $text" -ForegroundColor Cyan
}

function Assert-LastExit($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed with exit code $LASTEXITCODE." }
}

Write-Step 'Checking OpenAlex prerequisites'
if (-not (Test-Path $snapshot)) { throw "OpenAlex snapshot not found at $snapshot." }
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { throw 'dotnet was not found on PATH.' }
Write-Host "  snapshot : $snapshot"
Write-Host "  digest   : $digest"
Write-Host "  cloud    : openalex/abstracts"

if (-not (Test-Path $Python)) {
    if ($CheckOnly) {
        Write-Host "  would create sync environment at $Python"
    }
    else {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "Python environment missing at $Python and uv is not installed."
        }
        $venv = Split-Path -Parent (Split-Path -Parent $Python)
        & uv venv $venv --python 3.12
        Assert-LastExit 'uv venv'
    }
}

if ((Test-Path $Python) -and -not $env:legopds_clienthash -and -not $env:CLOUDPDS_CLIENT_HASH) {
    throw 'Set legopds_clienthash (or CLOUDPDS_CLIENT_HASH) before syncing.'
}

if (Test-Path $Python) {
    & $Python -c "import RemoteBlobStore" 2>$null
    if ($LASTEXITCODE -ne 0 -and -not $CheckOnly) {
        & uv pip install --python $Python "git+https://github.com/markusmobius/newsprinceton-pythoncloud"
        Assert-LastExit 'install RemoteBlobStore'
    }
}

Write-Step 'OpenAlex ingest'
if ($SkipIngest) {
    Write-Host '  -SkipIngest set'
}
elseif ((Test-Path $report) -and -not $Force) {
    $counts = (Get-Content $report -Raw | ConvertFrom-Json).counts
    Write-Host "  already built: $($counts.abstracts_written) abstracts from $($counts.works_read) works"
    Write-Host '  use -Force to rebuild'
}
elseif ($CheckOnly) {
    Write-Host "  would run OpenAlex.Ingest with $Threads readers"
}
else {
    if ($Force -and (Test-Path $digest)) { Remove-Item $digest -Recurse -Force }
    Push-Location $repo
    try {
        & dotnet run --project 'src\OpenAlex.Ingest' -c Release -- `
            --input $snapshot --out $digest --threads $Threads --shard-size $ShardSize
        Assert-LastExit 'OpenAlex.Ingest'
    }
    finally { Pop-Location }
}

if (-not $CheckOnly -and -not (Test-Path $report)) {
    throw "No ingest report at $report; refusing to sync."
}

Write-Step 'OpenAlex cloud sync'
if (-not (Test-Path $Python)) {
    Write-Host '  skipped because the sync environment does not exist yet'
    return
}

$savedMaxCores = $env:MAXCORES
if ($null -ne $savedMaxCores) { Remove-Item Env:MAXCORES -ErrorAction SilentlyContinue }
try {
    $arguments = @((Join-Path $PSScriptRoot 'openalex-cloudstore.py'), 'sync', '--local', $digest)
    if ($CheckOnly) { $arguments += '--check' }
    if ($Force) { $arguments += '--force' }
    & $Python @arguments
    Assert-LastExit 'OpenAlex cloud sync'
}
finally {
    if ($null -ne $savedMaxCores) { $env:MAXCORES = $savedMaxCores }
}

Write-Step 'Done'
Write-Host "  local digest : $digest"
Write-Host '  cloud path   : openalex/abstracts'