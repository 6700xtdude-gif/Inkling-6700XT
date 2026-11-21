<#
.SYNOPSIS
    Benchmark Inkling-2-Mini-UD-Q4_K_XL with llama-bench (loop-pinned -ot vs optional naive --n-cpu-moe 12).
#>
param(
    [string]$ModelPath = "C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf",
    [string]$OverrideTensor = "blk\.([0-5]|1[8-9]|2[0-3])\.ffn_.*_exps\.=CPU",
    [int]$PromptTokens = 512,
    [int]$GenTokens = 128,
    [string]$UbatchList = "256,512",
    [int]$Reps = 3,
    [switch]$CompareNaive
)

$BenchExe = "C:\llama.cpp\llama-bench.exe"
if (-not (Test-Path $BenchExe)) {
    Write-Error "llama-bench.exe not found at $BenchExe."
    exit 1
}
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found at $ModelPath."
    exit 1
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ReportsDir = Join-Path $RepoRoot "reports"
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutFile = Join-Path $ReportsDir "llama_bench_${Stamp}.txt"

Write-Host "=== Loop-Pinned (-ot '$OverrideTensor', -ub $UbatchList) ===" -ForegroundColor Cyan
$res1 = & $BenchExe -m $ModelPath -ngl 99 -ot $OverrideTensor -fa 1 -ctk q4_0 -ctv q4_0 -ub $UbatchList -p $PromptTokens -n $GenTokens -r $Reps 2>&1
$res1 | ForEach-Object { Write-Host $_ }

$logLines = @("=== Loop-Pinned (-ot $OverrideTensor, -ub $UbatchList) ===") + ($res1 | ForEach-Object { "$_" })

if ($CompareNaive) {
    Write-Host "`n=== Naive (--n-cpu-moe 12, -ub $UbatchList) ===" -ForegroundColor Yellow
    $res2 = & $BenchExe -m $ModelPath -ngl 99 --n-cpu-moe 12 -fa 1 -ctk q4_0 -ctv q4_0 -ub $UbatchList -p $PromptTokens -n $GenTokens -r $Reps 2>&1
    $res2 | ForEach-Object { Write-Host $_ }
    $logLines += @("", "=== Naive (--n-cpu-moe 12, -ub $UbatchList) ===") + ($res2 | ForEach-Object { "$_" })
}

$logLines | Set-Content -Path $OutFile -Encoding UTF8
Write-Host "`nSaved benchmark output to $OutFile"
