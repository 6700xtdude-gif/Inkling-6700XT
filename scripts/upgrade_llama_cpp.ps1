<#
.SYNOPSIS
    Upgrade C:\llama.cpp from b12427 to official mainline b12615 (win-vulkan-x64).
    Creates a rollback backup at C:\llama.cpp.bak_b12427 and preserves local helpers.
#>
$ErrorActionPreference = "Stop"

$Tag = "b12615"
$ZipName = "llama-b12615-bin-win-vulkan-x64.zip"
$Url = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/$ZipName"
$InstallDir = "C:\llama.cpp"
$BackupDir = "C:\llama.cpp.bak_b12427"
$WorkDir = Join-Path $env:TEMP "llama_upgrade_$Tag"
$ZipPath = Join-Path $WorkDir $ZipName
$ExtractDir = Join-Path $WorkDir "extracted"

# 1. Safety check: ensure no process is running out of C:\llama.cpp
$running = Get-Process | Where-Object {
    try { $_.Path -and $_.Path.StartsWith($InstallDir, [System.StringComparison]::OrdinalIgnoreCase) }
    catch { $false }
}
if ($running) {
    $names = ($running | ForEach-Object { "$($_.ProcessName) ($($_.Id))" }) -join ", "
    Write-Error "Aborting upgrade: running process(es) in ${InstallDir}: $names"
    exit 1
}

# 2. Download & extract
if (Test-Path $WorkDir) {
    Remove-Item -Recurse -Force $WorkDir
}
New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null

Write-Host "Downloading $ZipName from $Url ..."
& curl.exe -L --fail --retry 3 --retry-delay 3 --silent --show-error -o $ZipPath $Url
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download $Url (exit code $LASTEXITCODE)"
    exit 1
}
$zipBytes = (Get-Item $ZipPath).Length
Write-Host ("Downloaded {0} ({1:N1} MB)" -f $ZipName, ($zipBytes / 1MB))

Write-Host "Extracting to $ExtractDir ..."
Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force

# Support both flat archive layout and single-nested subfolder layout
$SourceRoot = $ExtractDir
if (-not (Test-Path (Join-Path $SourceRoot "llama-server.exe"))) {
    $subdirs = Get-ChildItem -Path $ExtractDir -Directory
    foreach ($sub in $subdirs) {
        if (Test-Path (Join-Path $sub.FullName "llama-server.exe")) {
            $SourceRoot = $sub.FullName
            break
        }
    }
}

foreach ($req in @("llama-server.exe", "ggml-vulkan.dll")) {
    if (-not (Test-Path (Join-Path $SourceRoot $req))) {
        Write-Error "Extracted archive missing required file '$req' under $SourceRoot"
        exit 1
    }
}

# 3. Backup existing C:\llama.cpp (first time only) and swap in b12615
if (Test-Path $InstallDir) {
    if (-not (Test-Path $BackupDir)) {
        Write-Host "Backing up existing $InstallDir -> $BackupDir ..."
        Move-Item -Path $InstallDir -Destination $BackupDir
    } else {
        Write-Host "Backup $BackupDir already exists; replacing $InstallDir ..."
        Remove-Item -Recurse -Force $InstallDir
    }
}

Write-Host "Installing $Tag into $InstallDir ..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Get-ChildItem -Path $SourceRoot -Force | Move-Item -Destination $InstallDir

# 4. Preserve local helper files from backup if missing
foreach ($helper in @("start-qwen7b.cmd", "start-qwen7b.ps1", "llama-connect.exe")) {
    $dst = Join-Path $InstallDir $helper
    $src = Join-Path $BackupDir $helper
    if (-not (Test-Path $dst) -and (Test-Path $src)) {
        Copy-Item -Path $src -Destination $dst
        Write-Host "Preserved local helper: $helper"
    }
}

# 5. Clean up temp staging
Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue

# 6. Verify version, llama.exe router binary, and Vulkan device detection
function Invoke-NativeQuietStderr {
    param([string]$ExePath, [string[]]$ArgList)
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $ExePath @ArgList 2>&1 | ForEach-Object { "$_" }
        return @{ ExitCode = $LASTEXITCODE; Output = $out }
    } finally {
        $ErrorActionPreference = $prevPref
    }
}

Write-Host "`n=== Verification: llama-server --version ==="
$vRes = Invoke-NativeQuietStderr -ExePath (Join-Path $InstallDir "llama-server.exe") -ArgList @("--version")
$vRes.Output | ForEach-Object { Write-Host $_ }
if ($vRes.ExitCode -ne 0) {
    Write-Error "llama-server.exe --version failed (exit $($vRes.ExitCode))"
    exit 1
}

$llamaExe = Join-Path $InstallDir "llama.exe"
if (-not (Test-Path $llamaExe)) {
    Write-Error "Missing $llamaExe (required by ENDO/SCRIPTS/start_llama_router.ps1)"
    exit 1
}
Write-Host "`n=== Verification: llama.exe --help (first 12 lines) ==="
$hRes = Invoke-NativeQuietStderr -ExePath $llamaExe -ArgList @("--help")
$hRes.Output | Select-Object -First 12 | ForEach-Object { Write-Host $_ }

Write-Host "`n=== Verification: llama-bench --list-devices ==="
$dRes = Invoke-NativeQuietStderr -ExePath (Join-Path $InstallDir "llama-bench.exe") -ArgList @("--list-devices")
$dRes.Output | ForEach-Object { Write-Host $_ }
if ($dRes.ExitCode -ne 0) {
    Write-Error "llama-bench.exe --list-devices failed (exit $($dRes.ExitCode))"
    exit 1
}

Write-Host "`n[OK] C:\llama.cpp upgraded cleanly to $Tag."
