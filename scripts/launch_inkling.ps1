<#
.SYNOPSIS
    Launch llama-server for Inkling-2-Mini-UD-Q4_K_XL on SANDBOX (RX 6700 XT 12GB + 64GB DDR4).
    Default profile: 32K context, q4_0/q4_0 quantized KV cache, loop-pinned -ot split (layers 6..17 in VRAM).
#>
param(
    [string]$ModelPath = "C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8088,
    [int]$Ctx = 32768,
    [string]$CacheTypeK = "q4_0",
    [string]$CacheTypeV = "q4_0",
    [int]$Parallel = 1,
    [string]$LoadMode = "none",
    [int]$Ubatch = 512,
    [int]$LogVerbosity = 3,
    [string]$ReasoningEffort = "low",
    [int]$ReasoningBudget = 1024,
    [string]$OverrideTensor = "blk\.([0-5]|1[8-9]|2[0-3])\.ffn_.*_exps\.=CPU"
)

$ServerExe = "C:\llama.cpp\llama-server.exe"
if (-not (Test-Path $ServerExe)) {
    Write-Error "llama-server.exe not found at $ServerExe. Run scripts/upgrade_llama_cpp.ps1 first."
    exit 1
}
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model file not found at $ModelPath. Run scripts/download_inkling.py first."
    exit 1
}

Write-Host "Starting Inkling-2-Mini on http://${BindHost}:$Port" -ForegroundColor Cyan
Write-Host "  Model:     $ModelPath"
Write-Host "  Ctx:       $Ctx | KV: ${CacheTypeK}/${CacheTypeV} | FlashAttn: on | np=$Parallel | lm=$LoadMode | ub=$Ubatch | mmproj: skipped"
Write-Host "  Reasoning: effort=$ReasoningEffort | budget=$ReasoningBudget | preserve=off"
Write-Host "  -ot:       $OverrideTensor"

& $ServerExe `
    -m $ModelPath `
    --host $BindHost `
    --port $Port `
    -ngl 99 `
    -ot $OverrideTensor `
    -c $Ctx `
    -np $Parallel `
    -lm $LoadMode `
    -ub $Ubatch `
    -lv $LogVerbosity `
    -fa on `
    --cache-type-k $CacheTypeK `
    --cache-type-v $CacheTypeV `
    --reasoning-effort $ReasoningEffort `
    --reasoning-budget $ReasoningBudget `
    --no-reasoning-preserve `
    --jinja `
    --temp 1.0 `
    --top-p 1.0 `
    --top-k 0 `
    --min-p 0.0

exit $LASTEXITCODE
