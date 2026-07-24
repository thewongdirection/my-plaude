<#
.SYNOPSIS
    Prepare an offline prerequisites bundle for plaude-local (PowerShell edition).

.DESCRIPTION
    PowerShell parity of tools/prepare_offline_bundle.py. Enumerates everything
    plaude-local needs (FFmpeg, Python wheels, CUDA runtime DLLs, Whisper models,
    optional diarization models), downloads it in advance, and zips it so a
    recipient can prep an offline / air-gapped machine by unzipping and following
    the generated INSTALL.md.

.EXAMPLE
    .\tools\Prepare-OfflineBundle.ps1 -List

.EXAMPLE
    .\tools\Prepare-OfflineBundle.ps1 -Models small,large-v3 -Include cuda,diarize -HfToken hf_xxx
#>
[CmdletBinding()]
param(
    [string[]]$Models = @('small', 'large-v3'),
    [ValidateSet('windows', 'linux', 'macos')]
    [string]$Platform = 'windows',
    [ValidateSet('cuda', 'diarize', 'deepfilter', 'resemble')]
    [string[]]$Include = @(),
    [string]$HfToken,
    [string]$ProjectRoot,
    [string]$Dest = 'plaude-local-offline',
    [string]$Zip = 'plaude-local-offline-bundle.zip',
    [switch]$NoZip,
    [switch]$List
)

$Script:FfmpegUrls = @{
    windows = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
    linux   = 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz'
    macos   = $null
}
$Script:IncludeWheels = @{
    cuda       = @('nvidia-cublas-cu12', 'nvidia-cudnn-cu12')
    diarize    = @('pyannote.audio>=3.1')
    deepfilter = @('deepfilternet')
    resemble   = @('resemble-enhance')
}
$Script:DiarizeRepos = @('pyannote/speaker-diarization-community-1', 'pyannote/speaker-diarization-3.1', 'pyannote/segmentation-3.0')

function Get-BundleManifest {
    # Pure enumeration (parity with Python build_manifest): returns an ordered
    # list of prerequisite objects {name, kind, source, target, note}.
    param(
        [string[]]$Models,
        [string]$Platform,
        [string[]]$Include,
        [bool]$WithDiarizeModels
    )
    $items = New-Object System.Collections.Generic.List[object]

    $ff = $Script:FfmpegUrls[$Platform]
    $items.Add([pscustomobject]@{
        name = 'FFmpeg (+ ffprobe)'; kind = $(if ($ff) { 'archive' } else { 'manual' })
        source = $(if ($ff) { $ff } else { 'brew install ffmpeg' }); target = 'ffmpeg/'
        note = 'required; decodes any input and denoises'
    })

    $wheels = [System.Collections.Generic.List[string]]::new()
    $wheels.Add('faster-whisper'); $wheels.Add('whisper-ctranslate2')
    foreach ($g in $Include) { foreach ($w in $Script:IncludeWheels[$g]) { $wheels.Add($w) } }
    $items.Add([pscustomobject]@{
        name = 'Python wheels'; kind = 'pip'; source = ($wheels -join ' '); target = 'wheels/'
        note = 'pip download into a local wheelhouse for --no-index installs'
    })

    foreach ($m in $Models) {
        $items.Add([pscustomobject]@{
            name = "Whisper model: $m"; kind = 'hf'; source = "Systran/faster-whisper-$m"
            target = 'hf/hub/'; note = 'transcription weights (faster-whisper / CTranslate2)'
        })
    }

    if ($WithDiarizeModels) {
        foreach ($repo in $Script:DiarizeRepos) {
            $items.Add([pscustomobject]@{
                name = "Diarization model: $repo"; kind = 'hf-gated'; source = $repo
                target = 'hf/hub/'; note = 'GATED: needs an HF token + accepted model terms'
            })
        }
    }
    return $items.ToArray()
}

function Get-ArchiveResource {
    param([string]$Url, [string]$Dest)
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $out = Join-Path $Dest ($Url.Split('/')[-1])
    Write-Host "  downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $out -UseBasicParsing
    Write-Host ("  saved {0} ({1} MB)" -f (Split-Path $out -Leaf), [int]((Get-Item $out).Length / 1MB))
}

function Get-PipResource {
    param([string[]]$Packages, [string]$Dest)
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    & python -m pip download -d $Dest @Packages
    if ($LASTEXITCODE -ne 0) { throw "pip download failed (exit $LASTEXITCODE)" }
}

function Get-SelfWheel {
    param([string]$ProjectRoot, [string]$Dest)
    if (-not (Test-Path (Join-Path $ProjectRoot 'pyproject.toml'))) {
        Write-Host "  (skip self wheel: no pyproject.toml at $ProjectRoot)"; return
    }
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    & python -m pip wheel $ProjectRoot -w $Dest
    if ($LASTEXITCODE -ne 0) { throw "pip wheel failed (exit $LASTEXITCODE)" }
}

function Get-HfResource {
    param([string]$Repo, [string]$CacheDir, [string]$Token)
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
    $py = @'
import sys
from huggingface_hub import snapshot_download
repo, cache, token = sys.argv[1], sys.argv[2], (sys.argv[3] or None)
snapshot_download(repo_id=repo, cache_dir=cache, token=token)
'@
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('hfdl_' + [System.IO.Path]::GetRandomFileName() + '.py')
    Set-Content -LiteralPath $tmp -Value $py -Encoding UTF8
    try {
        Write-Host "  snapshot $Repo -> $CacheDir"
        & python $tmp $Repo $CacheDir "$Token"
        if ($LASTEXITCODE -ne 0) { throw "snapshot_download failed (exit $LASTEXITCODE)" }
    } finally { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
}

function Write-InstallDoc {
    param([string]$Dest, $Manifest, [string]$Platform)
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('# plaude-local - offline prerequisites bundle')
    $lines.Add('')
    $lines.Add("Platform target: **$Platform**. Everything below is included so you can install offline.")
    $lines.Add(''); $lines.Add('## Contents'); $lines.Add('')
    foreach ($i in $Manifest) { $lines.Add("- **$($i.name)** -> ``$($i.target)`` ($($i.note))") }
    $lines.Add(''); $lines.Add('## Install steps'); $lines.Add('')
    $lines.Add('1. Unzip somewhere permanent, e.g. C:\plaude-local-offline.')
    $lines.Add('2. FFmpeg: unzip ffmpeg\ and add its bin\ folder to PATH.')
    $lines.Add('3. Python packages (no internet):')
    $lines.Add('   pip install --no-index --find-links wheels plaude-local')
    $lines.Add('4. Models: set HF_HOME to the bundle''s hf folder, then pass --offline / -Offline.')
    $lines.Add('5. Verify: plaude-local --check  (or .\Plaude-Local.ps1 -Check)')
    $lines.Add('')
    $lines.Add('Windows note: installing pyannote.audio pulls PyTorch, whose nested files can')
    $lines.Add('exceed the 260-char path limit (WinError 206). Enable long paths (admin) or use')
    $lines.Add('a virtual environment created at a short path such as C:\v.')
    Set-Content -LiteralPath (Join-Path $Dest 'INSTALL.md') -Value ($lines -join "`n") -Encoding UTF8
}

function New-BundleZip {
    param([string]$Staging, [string]$ZipPath)
    if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
    $parent = Split-Path $ZipPath -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    Compress-Archive -Path (Join-Path $Staging '*') -DestinationPath $ZipPath -Force
    return $ZipPath
}

function Invoke-Main {
    # Emits data to the success stream (e.g. -List JSON); the exit code is carried
    # in $Script:BundleExit so the caller can `exit` WITHOUT capturing (and thus
    # swallowing) that output. Never `return` a value here.
    $Script:BundleExit = 0
    $wantDiar = $Include -contains 'diarize'
    $manifest = Get-BundleManifest -Models $Models -Platform $Platform -Include $Include -WithDiarizeModels $wantDiar

    if ($List) { $manifest | ConvertTo-Json -Depth 4; return }

    $token = if ($HfToken) { $HfToken } else { $env:HF_TOKEN }
    if ($wantDiar -and -not $token) {
        Write-Error 'Include diarize needs an HF token (-HfToken or HF_TOKEN) and accepted model terms.'
        $Script:BundleExit = 2
        return
    }

    $staging = (New-Item -ItemType Directory -Force -Path $Dest).FullName
    Write-Host "Staging into: $staging"
    foreach ($item in $manifest) {
        Write-Host "[$($item.kind)] $($item.name)"
        $target = Join-Path $staging ($item.target.TrimEnd('/'))
        try {
            switch ($item.kind) {
                'archive'  { Get-ArchiveResource -Url $item.source -Dest $target }
                'manual'   { Write-Host "  MANUAL: $($item.source)" }
                'pip'      { Get-PipResource -Packages ($item.source -split ' ') -Dest $target }
                'hf'       { Get-HfResource -Repo $item.source -CacheDir $target -Token $token }
                'hf-gated' { Get-HfResource -Repo $item.source -CacheDir $target -Token $token }
            }
        } catch { Write-Warning "failed to fetch $($item.name): $($_.Exception.Message)" }
    }

    $root = if ($ProjectRoot) { $ProjectRoot }
            elseif ($PSScriptRoot) { Split-Path $PSScriptRoot -Parent }
            else { (Get-Location).Path }
    Get-SelfWheel -ProjectRoot $root -Dest (Join-Path $staging 'wheels')
    Write-InstallDoc -Dest $staging -Manifest $manifest -Platform $Platform
    ($manifest | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath (Join-Path $staging 'manifest.json') -Encoding UTF8

    if ($NoZip) { Write-Host "Done. Staged (not zipped) at $staging"; return }
    $zipFull = if ([System.IO.Path]::IsPathRooted($Zip)) { $Zip } else { Join-Path (Get-Location).Path $Zip }
    $zipPath = New-BundleZip -Staging $staging -ZipPath $zipFull
    Write-Host "Done. Bundle: $zipPath"
}

# Only run when invoked directly - NOT when dot-sourced by the Pester tests.
if ($MyInvocation.InvocationName -ne '.') {
    $Script:BundleExit = 0
    Invoke-Main            # emits output uncaptured so -List JSON reaches stdout
    exit $Script:BundleExit
}
