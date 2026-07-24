#requires -Version 5.1
<#
.SYNOPSIS
    plaude-local (PowerShell edition) - local, offline speech-to-text.

.DESCRIPTION
    A Windows-first PowerShell port of the Python plaude-local tool, kept at
    feature parity with it. Transcribes audio/video in ANY format FFmpeg can
    decode, fully offline, on an NVIDIA GPU (CUDA) or CPU.

    Pipeline (identical to the Python version):
        1. denoise / normalize     (FFmpeg, or optional DeepFilterNet)
        2. transcribe              (whisper-ctranslate2 = faster-whisper engine)
        3. speaker diarization      (optional - pyannote / whisperx)
        4. summarize               (optional - local LLM via Ollama / llama.cpp)
    Output is always written as UTF-8 so Chinese/Japanese/Korean and any
    non-Latin script round-trip correctly.

    External tools (see .\README.md):
      * ffmpeg + ffprobe            (required)
      * whisper-ctranslate2         (required; pip install whisper-ctranslate2)
      * deepFilter                  (optional; -Denoise deepfilter)
      * Ollama or llama.cpp server  (optional; -Summarize)

.EXAMPLE
    .\Plaude-Local.ps1 note.mp3

.EXAMPLE
    .\Plaude-Local.ps1 meeting.m4a -Diarize -Summarize -Format txt

.EXAMPLE
    .\Plaude-Local.ps1 -Check
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$InputFile,

    [Alias('o')]
    [string]$Output,

    [ValidateSet('txt', 'srt', 'vtt', 'json')]
    [string]$Format = 'txt',

    [Alias('Doctor')]
    [switch]$Check,

    # model / hardware
    [string]$Model = 'large-v3',
    [ValidateSet('auto', 'cuda', 'cpu')]
    [string]$Device = 'auto',
    [string]$ComputeType = 'auto',
    [string]$Language,
    [int]$BeamSize = 5,
    [switch]$NoVad,
    [string]$ModelDir,
    [switch]$Offline,

    # audio preprocessing
    [ValidateSet('ffmpeg', 'deepfilter', 'none')]
    [string]$Denoise = 'ffmpeg',
    [ValidateSet('none', 'speech', 'strong', 'resemble')]
    [string]$Enhance = 'none',
    [double]$Gain = 0.0,
    [string]$KeepClean,

    # diarization
    [switch]$Diarize,
    [ValidateSet('pyannote', 'whisperx')]
    [string]$DiarizeBackend = 'pyannote',
    [string]$HfToken,
    [string]$DiarizeModel,
    [int]$NumSpeakers,
    [int]$MinSpeakers,
    [int]$MaxSpeakers,

    # summarization
    [switch]$Summarize,
    [ValidateSet('auto', 'ollama', 'llamacpp')]
    [string]$SummarizeBackend = 'auto',
    [string]$SummarizeModel,
    [string]$SummarizeUrl,
    [string]$SummaryOutput,
    [int]$SummarizeMaxChars = 8000,

    # recording quality
    [switch]$AssessOnly,
    [ValidateSet('warn', 'skip', 'fail')]
    [string]$OnBad = 'warn',
    [double]$MinSpeech = 0.15,
    [double]$MaxCompression = 2.4,
    [double]$MinLogprob = -1.0,
    [double]$MaxNoSpeech = 0.6,

    # prerequisites / provisioning
    [string]$FfmpegLocation,
    [switch]$InstallMissing,
    [switch]$NoProvision,

    # output behavior
    [Alias('y', 'Overwrite')]
    [switch]$Yes,
    [switch]$Quiet,
    [switch]$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- constants (parity with the Python version) ---------------------------- #
$Script:TargetSr = 16000
$Script:FfmpegDenoiseChain = 'highpass=f=90,lowpass=f=7500,afftdn=nf=-25,dynaudnorm'
$Script:EnhanceChains = @{
    speech = 'highpass=f=80,speechnorm=e=6.25:r=0.0005:l=1,loudnorm=I=-16:TP=-1.5:LRA=11'
    strong = 'highpass=f=80,acompressor=threshold=-18dB:ratio=3:attack=20:release=250,' +
             'equalizer=f=3000:width_type=q:w=1.5:g=4,speechnorm=e=12.5:r=0.0005:l=1,' +
             'loudnorm=I=-16:TP=-1.5:LRA=11'
}
$Script:DefaultOllamaUrl = 'http://localhost:11434'
$Script:DefaultLlamacppUrl = 'http://localhost:8080'
$Script:DefaultOllamaModel = 'llama3.1'
$Script:OverwriteTimeoutSeconds = 10
$Script:ToolVersion = '0.1.0'
$Script:QualityFixed = @{ NearSilentDb = -50.0; MostlySilence = 0.85 }

# Emit UTF-8 to the console so CJK shows correctly regardless of code page.
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
function Write-Log {
    param([string]$Message)
    if (-not $Quiet) { [Console]::Error.WriteLine($Message) }
}

function Write-ErrLine {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Write-Utf8File {
    param([string]$Path, [string]$Text)
    # UTF-8 without BOM.
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function Format-Db {
    param([double]$Db)
    # Always use '.' as the decimal separator so the ffmpeg 'volume=<n>dB'
    # filter parses on every locale (parity with Python's f-string).
    return $Db.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-DefaultOutput {
    param([string]$Fmt)
    return "output.$Fmt"
}

# --------------------------------------------------------------------------- #
# Overwrite confirmation with a 10-second timeout (parity with Python)
# --------------------------------------------------------------------------- #
function Read-LineWithTimeout {
    param([int]$TimeoutSeconds)
    # Run Console.ReadLine on a runspace in THIS process so it shares the
    # console; wait up to the timeout, else return $null.
    $ps = [PowerShell]::Create()
    [void]$ps.AddScript('[Console]::ReadLine()')
    $async = $ps.BeginInvoke()
    try {
        if ($async.AsyncWaitHandle.WaitOne([TimeSpan]::FromSeconds($TimeoutSeconds))) {
            $result = $ps.EndInvoke($async)
            if ($result -and $result.Count -gt 0) { return [string]$result[0] }
            return ''
        }
        return $null  # timed out
    } finally {
        try { $ps.Stop() } catch { }
        $ps.Dispose()
    }
}

function Confirm-Overwrite {
    param([string]$Path)
    if ($Yes -or -not (Test-Path -LiteralPath $Path)) { return $true }
    if ([Console]::IsInputRedirected) {
        Write-Log "note: $Path exists; overwriting (non-interactive session)."
        return $true
    }
    [Console]::Error.Write("$Path already exists. Overwrite? [Y/n] (auto-overwrite in $($Script:OverwriteTimeoutSeconds)s): ")
    $answer = Read-LineWithTimeout -TimeoutSeconds $Script:OverwriteTimeoutSeconds
    [Console]::Error.WriteLine('')
    if ($null -eq $answer) { return $true }             # timed out -> overwrite
    switch ($answer.Trim().ToLower()) {
        'y'   { return $true }
        'yes' { return $true }
        'n'   { return $false }
        'no'  { return $false }
        default { return $true }                        # blank/other -> default
    }
}

# --------------------------------------------------------------------------- #
# FFmpeg audio: probe + denoise
# --------------------------------------------------------------------------- #
function Get-AudioCodec {
    param([string]$Path)
    if (-not (Test-Command 'ffprobe')) { return $null }
    $out = & ffprobe -v error -select_streams a:0 `
        -show_entries stream=codec_name `
        -of default=nokey=1:noprint_wrappers=1 -- $Path 2>$null
    $codec = ($out | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($codec)) { return $null }
    return $codec
}

$Script:FfmpegDownloadWindows = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'

function Register-FfmpegPath {
    param([string]$Location)
    $dir = if (Test-Path -LiteralPath $Location -PathType Leaf) { Split-Path $Location -Parent } else { $Location }
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) { return $false }
    $hasFf = (Test-Path (Join-Path $dir 'ffmpeg.exe')) -or (Test-Path (Join-Path $dir 'ffmpeg'))
    $hasPr = (Test-Path (Join-Path $dir 'ffprobe.exe')) -or (Test-Path (Join-Path $dir 'ffprobe'))
    if (-not ($hasFf -and $hasPr)) { return $false }
    $env:PATH = $dir + [System.IO.Path]::PathSeparator + $env:PATH
    # Re-verify they actually resolve now (parity with Python's have_* recheck).
    return ((Test-Command 'ffmpeg') -and (Test-Command 'ffprobe'))
}

function Install-Ffmpeg {
    # Prefer winget when present; otherwise download a self-contained build.
    if (Test-Command 'winget') {
        & winget install --id Gyan.FFmpeg -e --source winget `
            --accept-package-agreements --accept-source-agreements 2>&1 |
            ForEach-Object { Write-Log ([string]$_) }
        if ((Test-Command 'ffmpeg') -and (Test-Command 'ffprobe')) { return $true }
    }
    $target = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'plaude-local\ffmpeg'
    try {
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        $zip = Join-Path $target 'ffmpeg.zip'
        Write-Log "downloading FFmpeg from $($Script:FfmpegDownloadWindows) ..."
        Invoke-WebRequest -Uri $Script:FfmpegDownloadWindows -OutFile $zip
        Expand-Archive -LiteralPath $zip -DestinationPath $target -Force
        Remove-Item -LiteralPath $zip -ErrorAction SilentlyContinue
        $exe = Get-ChildItem -LiteralPath $target -Recurse -Filter 'ffmpeg.exe' |
            Select-Object -First 1
        if ($exe -and (Register-FfmpegPath -Location $exe.DirectoryName)) {
            Write-Log "installed FFmpeg to $target"
            return $true
        }
    } catch {
        Write-Log "automatic FFmpeg install failed: $($_.Exception.Message)"
    }
    return $false
}

function Confirm-Ffmpeg {
    if (Test-Command 'ffmpeg') {
        if (-not (Test-Command 'ffprobe') -and -not $Quiet) {
            Write-ErrLine 'note: ffprobe not found (it ships with FFmpeg); stream checks and some quality metrics will be limited.'
        }
        return $true
    }
    Write-ErrLine 'warning: FFmpeg (ffmpeg + ffprobe) was not found on PATH. It is required to decode audio.'

    if ($FfmpegLocation) {
        if (Register-FfmpegPath -Location $FfmpegLocation) { Write-ErrLine "using FFmpeg from $FfmpegLocation"; return $true }
        Write-ErrLine "error: no usable ffmpeg/ffprobe found at '$FfmpegLocation'."
    }
    if ($InstallMissing) {
        Write-ErrLine 'attempting to download and install FFmpeg ...'
        if (Install-Ffmpeg) { return $true }
        Write-ErrLine 'error: automatic FFmpeg installation did not succeed.'
        return $false
    }
    $interactive = -not [Console]::IsInputRedirected
    if ($NoProvision -or (-not $interactive)) {
        Write-ErrLine 'aborting: FFmpeg is unavailable. Install it (winget/apt/brew or https://ffmpeg.org/download.html), pass -FfmpegLocation PATH, or re-run with -InstallMissing.'
        return $false
    }
    while ($true) {
        $choice = (Read-Host 'Choose: [i]nstall automatically, provide a [p]ath, or [a]bort?').Trim().ToLower()
        switch ($choice) {
            { $_ -in 'i', 'install' } {
                if (Install-Ffmpeg) { return $true }
                Write-ErrLine 'automatic installation failed; try providing a path instead.'
            }
            { $_ -in 'p', 'path' } {
                $loc = (Read-Host 'Enter the folder containing ffmpeg/ffprobe (or the ffmpeg binary)').Trim()
                if ($loc -and (Register-FfmpegPath -Location $loc)) { Write-ErrLine "using FFmpeg from $loc"; return $true }
                Write-ErrLine 'that path did not contain a usable ffmpeg + ffprobe.'
            }
            { $_ -in 'a', 'abort', 'n', 'no', '' } {
                Write-ErrLine 'aborting at user request: required FFmpeg not provided.'
                return $false
            }
            default { Write-ErrLine "unrecognized choice '$choice'; please enter i, p, or a." }
        }
    }
}

function Get-MediaDuration {
    param([string]$Path)
    if (-not (Test-Command 'ffprobe')) { return $null }
    $out = & ffprobe -v error -show_entries format=duration `
        -of default=nokey=1:noprint_wrappers=1 -- $Path 2>$null
    $val = ($out | Out-String).Trim()
    $d = 0.0
    if ([double]::TryParse($val, [ref]$d)) { return $d }
    return $null
}

function ConvertFrom-FfmpegLevels {
    # Pure parser (unit-tested): ffmpeg volumedetect/silencedetect stderr -> stats.
    param([string]$Text, $Duration)
    $stats = @{ mean_volume_db = $null; max_volume_db = $null; silence_ratio = $null }
    $m = [regex]::Match($Text, 'mean_volume:\s*(-?\d+(?:\.\d+)?) dB')
    if ($m.Success) { $stats.mean_volume_db = [double]$m.Groups[1].Value }
    $m = [regex]::Match($Text, 'max_volume:\s*(-?\d+(?:\.\d+)?) dB')
    if ($m.Success) { $stats.max_volume_db = [double]$m.Groups[1].Value }
    $silence = 0.0
    foreach ($mm in [regex]::Matches($Text, 'silence_duration:\s*(\d+(?:\.\d+)?)')) {
        $silence += [double]$mm.Groups[1].Value
    }
    if ($Duration -and $Duration -gt 0) {
        $stats.silence_ratio = [Math]::Max(0.0, [Math]::Min(1.0, $silence / $Duration))
    }
    return $stats
}

function Get-AudioStats {
    param([string]$Path)
    $err = & ffmpeg -hide_banner -nostats -i $Path `
        -af 'volumedetect,silencedetect=noise=-30dB:d=0.5' -f null - 2>&1 |
        Out-String
    return ConvertFrom-FfmpegLevels -Text $err -Duration (Get-MediaDuration $Path)
}

function Invoke-Ffmpeg {
    param([string[]]$FfArgs)
    $full = @('-hide_banner', '-loglevel', 'error', '-y') + $FfArgs
    & ffmpeg @full 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg failed (exit $LASTEXITCODE) for: ffmpeg $($full -join ' ')"
    }
}

function ConvertTo-Wav {
    param([string]$Src, [string]$Dst, [string]$Filters)
    $a = @('-i', $Src)
    if ($Filters) { $a += @('-af', $Filters) }
    $a += @('-ac', '1', '-ar', "$($Script:TargetSr)", '-c:a', 'pcm_s16le', $Dst)
    Invoke-Ffmpeg -FfArgs $a
}

function Invoke-DeepFilter {
    param([string]$Src, [string]$Dst)
    if (-not (Test-Command 'deepFilter')) {
        throw "DeepFilterNet's 'deepFilter' command not found. Install it (pip install deepfilternet) or use -Denoise ffmpeg / -Denoise none."
    }
    $tmp48 = [System.IO.Path]::ChangeExtension($Dst, '.df48.wav')
    ConvertTo-Wav -Src $Src -Dst $tmp48 -Filters $null
    & deepFilter $tmp48 -o (Split-Path $tmp48 -Parent) 2>&1 | ForEach-Object { Write-Log ([string]$_) }
    if ($LASTEXITCODE -ne 0) { throw "deepFilter failed (exit $LASTEXITCODE)." }
    # deepFilter writes <name>_DeepFilterNet3.wav next to input; find newest wav.
    $enhanced = Get-ChildItem -LiteralPath (Split-Path $tmp48 -Parent) -Filter '*.wav' |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    ConvertTo-Wav -Src $enhanced.FullName -Dst $Dst -Filters $null
    Remove-Item -LiteralPath $tmp48 -ErrorAction SilentlyContinue
}

function Get-EnhanceFilters {
    param([string]$Mode, [double]$GainDb)
    $parts = @()
    if ($Script:EnhanceChains.ContainsKey($Mode)) { $parts += $Script:EnhanceChains[$Mode] }
    if ($GainDb -ne 0.0) { $parts += "volume=$(Format-Db $GainDb)dB" }
    if ($parts.Count -eq 0) { return $null }
    return ($parts -join ',')
}

function Invoke-ResembleEnhance {
    param([string]$Src, [string]$Dst)
    if (-not (Test-Command 'resemble-enhance')) {
        throw "Resemble-Enhance's 'resemble-enhance' command not found. Install it (pip install resemble-enhance) or use -Enhance speech / -Enhance strong / -Enhance none."
    }
    # The resemble-enhance CLI processes a directory of wavs into an output dir.
    $inDir = Join-Path (Split-Path $Dst -Parent) 're_in'
    $outDir = Join-Path (Split-Path $Dst -Parent) 're_out'
    New-Item -ItemType Directory -Path $inDir, $outDir -Force | Out-Null
    Copy-Item -LiteralPath $Src -Destination (Join-Path $inDir 'audio.wav') -Force
    & resemble-enhance $inDir $outDir 2>&1 | ForEach-Object { Write-Log ([string]$_) }
    if ($LASTEXITCODE -ne 0) { throw "resemble-enhance failed (exit $LASTEXITCODE)." }
    $enhanced = Get-ChildItem -LiteralPath $outDir -Filter '*.wav' | Select-Object -First 1
    if (-not $enhanced) { throw 'resemble-enhance produced no output.' }
    ConvertTo-Wav -Src $enhanced.FullName -Dst $Dst -Filters $null  # 16 kHz mono
}

function Invoke-Prepare {
    param([string]$Src, [string]$WorkDir)
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Src)
    $final = Join-Path $WorkDir ($stem + '.prepared.wav')
    $needEnhance = ($Enhance -ne 'none') -or ($Gain -ne 0.0)
    $denoiseDst = if ($needEnhance) { Join-Path $WorkDir ($stem + '.denoised.wav') } else { $final }

    switch ($Denoise) {
        'none'       { ConvertTo-Wav -Src $Src -Dst $denoiseDst -Filters $null }
        'ffmpeg'     { ConvertTo-Wav -Src $Src -Dst $denoiseDst -Filters $Script:FfmpegDenoiseChain }
        'deepfilter' { Invoke-DeepFilter -Src $Src -Dst $denoiseDst }
    }

    if (-not $needEnhance) { return $final }

    if ($Enhance -eq 'resemble') {
        Invoke-ResembleEnhance -Src $denoiseDst -Dst $final
        if ($Gain -ne 0.0) {
            $gained = Join-Path $WorkDir ($stem + '.gained.wav')
            ConvertTo-Wav -Src $final -Dst $gained -Filters "volume=$(Format-Db $Gain)dB"
            Move-Item -LiteralPath $gained -Destination $final -Force
        }
    } else {
        ConvertTo-Wav -Src $denoiseDst -Dst $final -Filters (Get-EnhanceFilters -Mode $Enhance -GainDb $Gain)
    }
    return $final
}

# --------------------------------------------------------------------------- #
# Transcription via whisper-ctranslate2 (same faster-whisper/CTranslate2 engine)
# --------------------------------------------------------------------------- #
function Set-HfOffline {
    # Force Hugging Face model loads to use only the local cache: HF_HUB_OFFLINE
    # (whisper-ctranslate2 / pyannote weights) + TRANSFORMERS_OFFLINE (whisperx).
    # The "download once online, then run fully offline" flow. Parity with the
    # Python cli.apply_offline(). No-op when disabled.
    param([bool]$Enabled)
    if (-not $Enabled) { return }
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
}

function Resolve-Device {
    if ($Device -ne 'auto') { return $Device }
    # nvidia-smi present and succeeds -> assume CUDA available.
    if (Test-Command 'nvidia-smi') {
        & nvidia-smi *> $null
        if ($LASTEXITCODE -eq 0) { return 'cuda' }
    }
    return 'cpu'
}

function Resolve-ComputeType {
    param([string]$Dev)
    if ($ComputeType -ne 'auto') { return $ComputeType }
    if ($Dev -eq 'cuda') { return 'float16' } else { return 'int8' }
}

function Resolve-PythonExe {
    foreach ($name in @('python', 'py')) {
        if (Test-Command $name) { return $name }
    }
    return $null
}

function Get-CudaBootstrap {
    # A Python one-shot that registers the pip 'nvidia-*-cu12' wheel bin dirs
    # (cublas64_12.dll / cudnn64_9.dll ...) with os.add_dll_directory, then runs
    # the whisper-ctranslate2 entry point. Needed because a child process cannot
    # see those DLLs via PATH under Python 3.8+'s restricted DLL search. Parity
    # with the Python package's transcribe.add_cuda_dll_directories().
    return @'
import os, sys, glob, importlib.util
try:
    s = importlib.util.find_spec("nvidia")
    roots = list(s.submodule_search_locations) if s and s.submodule_search_locations else []
    for root in roots:
        for b in glob.glob(os.path.join(root, "*", "bin")):
            if os.path.isdir(b):
                try:
                    os.add_dll_directory(b)
                except OSError:
                    pass
except Exception:
    pass
from whisper_ctranslate2.whisper_ctranslate2 import main
main()
'@
}

function Invoke-Transcribe {
    param([string]$AudioPath, [string]$WorkDir, [string]$Dev, [string]$Compute, [string]$Token)

    # Always request 'all' so we get JSON (per-segment quality metrics) and a
    # plain-text copy (for summarization) alongside the requested format.
    $a = @(
        $AudioPath,
        '--model', $Model,
        '--device', $Dev,
        '--compute_type', $Compute,
        '--output_dir', $WorkDir,
        '--output_format', 'all',
        '--beam_size', "$BeamSize",
        # Always pass VAD explicitly (parity with Python's vad_filter=not no_vad).
        '--vad_filter', $(if ($NoVad) { 'False' } else { 'True' })
    )
    if ($Language) { $a += @('--language', $Language) }
    # whisper-ctranslate2 enables speaker diarization by the mere presence of an
    # HF token (there is no --speaker_diarization / --*_speakers flag).
    if ($Diarize -and $Token) { $a += @('--hf_token', $Token) }

    # Route the tool's own stdout/stderr to our log so it does NOT become part
    # of this function's return value (PowerShell captures success-stream output).
    if ($Dev -eq 'cuda') {
        # A child process can't see the nvidia-*-cu12 wheel CUDA DLLs via PATH
        # (Python 3.8+ restricts the DLL search), so run whisper-ctranslate2
        # through a bootstrap that registers those dirs first (parity with the
        # Python entry point). Requires 'python' on PATH. The bootstrap is
        # written to a file (not `python -c`) to avoid PowerShell mangling the
        # embedded quotes/newlines when it builds the native command line.
        $py = Resolve-PythonExe
        if (-not $py) {
            throw "python (or py) must be on PATH to run whisper-ctranslate2 on the GPU."
        }
        $bootstrap = Join-Path $WorkDir '_cuda_bootstrap.py'
        Write-Utf8File -Path $bootstrap -Text (Get-CudaBootstrap)
        & $py $bootstrap @a 2>&1 | ForEach-Object { Write-Log ([string]$_) }
    } else {
        & whisper-ctranslate2 @a 2>&1 | ForEach-Object { Write-Log ([string]$_) }
    }
    if ($LASTEXITCODE -ne 0) { throw "whisper-ctranslate2 failed (exit $LASTEXITCODE)." }

    $base = [System.IO.Path]::GetFileNameWithoutExtension($AudioPath)
    $produced = Join-Path $WorkDir "$base.$Format"
    if (-not (Test-Path -LiteralPath $produced)) {
        throw "expected transcript not found at $produced"
    }
    return $produced
}

# --------------------------------------------------------------------------- #
# Summarization (Ollama / llama.cpp) - parity with plaude_local/summarize.py
# --------------------------------------------------------------------------- #
function Test-OllamaUp {
    param([string]$Url)
    try { Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/api/tags" -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}
function Test-LlamacppUp {
    param([string]$Url)
    try { Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/health" -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}
function Get-SummBackend {
    param([string]$OllamaUrl, [string]$LlamacppUrl)
    if (Test-OllamaUp $OllamaUrl) { return 'ollama' }
    if (Test-LlamacppUp $LlamacppUrl) { return 'llamacpp' }
    return $null
}

function Build-SummaryPrompt {
    param([string]$Text, [switch]$Combine)
    if ($Combine) {
        $head = 'You are a helpful assistant. Below are partial summaries of a longer transcript. Combine them into one concise final summary with key topics, decisions, and action items as bullet points.'
        $label = 'Partial summaries'
    } else {
        $head = 'You are a helpful assistant. Summarize the following transcript into concise bullet points capturing the key topics, decisions, and any action items. Preserve the original language of the transcript.'
        $label = 'Transcript'
    }
    return "$head`n`n$label`:`n$Text`n`nSummary:"
}

function Split-IntoChunks {
    param([string]$Text, [int]$MaxChars)
    $MaxChars = [Math]::Max($MaxChars, 1)
    if ($Text.Length -le $MaxChars) {
        if ($Text.Trim()) { return , @($Text) } else { return , @() }
    }
    $chunks = New-Object System.Collections.Generic.List[string]
    $remaining = $Text
    while ($remaining.Length -gt $MaxChars) {
        $window = $remaining.Substring(0, $MaxChars)
        $cut = $window.LastIndexOf("`n")
        # Floor division to match Python's `max_chars // 2` exactly. [int] would
        # apply banker's rounding (e.g. [int](7/2)=4), diverging on odd sizes.
        if ($cut -lt [Math]::Floor($MaxChars / 2)) { $cut = $MaxChars }
        if ($cut -lt 1) { $cut = 1 }
        $piece = $remaining.Substring(0, $cut).Trim()
        if ($piece) { [void]$chunks.Add($piece) }
        $remaining = $remaining.Substring($cut)
    }
    if ($remaining.Trim()) { [void]$chunks.Add($remaining.Trim()) }
    return , $chunks.ToArray()
}

function Invoke-LlmCall {
    param([string]$Prompt, [string]$Backend, [string]$Model, [string]$Url)
    # Encode the JSON body as UTF-8 bytes so non-Latin (CJK) transcript text is
    # sent correctly regardless of the default request encoding.
    if ($Backend -eq 'ollama') {
        $json = @{ model = $Model; prompt = $Prompt; stream = $false } | ConvertTo-Json
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
        $resp = Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/api/generate" -Method Post `
            -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec 120
        if (-not $resp.response) { throw 'ollama returned no text.' }
        return ([string]$resp.response).Trim()
    } else {
        $json = @{ prompt = $Prompt; n_predict = 512; temperature = 0.2; stream = $false } | ConvertTo-Json
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
        $resp = Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/completion" -Method Post `
            -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec 120
        if (-not $resp.content) { throw 'llama.cpp returned no text.' }
        return ([string]$resp.content).Trim()
    }
}

function Invoke-Summarize {
    param(
        [string]$Text, [string]$Backend, [string]$Model, [string]$Url, [int]$MaxChars,
        [int]$Depth = 0
    )
    $Text = $Text.Trim()
    if (-not $Text) { throw 'nothing to summarize: the transcript is empty.' }
    $MaxChars = [Math]::Max($MaxChars, 1)

    $chunks = Split-IntoChunks -Text $Text -MaxChars $MaxChars
    if ($chunks.Count -le 1) {
        return Invoke-LlmCall -Prompt (Build-SummaryPrompt -Text $chunks[0]) -Backend $Backend -Model $Model -Url $Url
    }
    $partials = foreach ($c in $chunks) {
        Invoke-LlmCall -Prompt (Build-SummaryPrompt -Text $c) -Backend $Backend -Model $Model -Url $Url
    }
    $combined = ($partials -join "`n`n")
    # Map-reduce (parity with Python's summarize_text): if the combined partial
    # summaries still exceed the budget, reduce again - bounded to depth 3 so it
    # always terminates.
    if ($combined.Length -le $MaxChars -or $Depth -ge 3) {
        return Invoke-LlmCall -Prompt (Build-SummaryPrompt -Text $combined -Combine) -Backend $Backend -Model $Model -Url $Url
    }
    return Invoke-Summarize -Text $combined -Backend $Backend -Model $Model -Url $Url -MaxChars $MaxChars -Depth ($Depth + 1)
}

# --------------------------------------------------------------------------- #
# Recording quality assessment - parity with plaude_local/quality.py
# --------------------------------------------------------------------------- #
function Get-Thresholds {
    return @{
        MinSpeech = $MinSpeech; MaxCompression = $MaxCompression
        MinLogprob = $MinLogprob; MaxNoSpeech = $MaxNoSpeech
        NearSilentDb = $Script:QualityFixed.NearSilentDb
        MostlySilence = $Script:QualityFixed.MostlySilence
    }
}

function _MeanOrNull {
    param([object[]]$Values)
    $nums = @($Values | Where-Object { $null -ne $_ })
    if ($nums.Count -eq 0) { return $null }
    return ($nums | Measure-Object -Average).Average
}

function _Prop {
    # Null-safe property read (parity with Python's dict.get): returns $null for
    # an absent property instead of throwing under Set-StrictMode -Version Latest.
    param($Obj, [string]$Name)
    $p = $Obj.PSObject.Properties[$Name]
    if ($p) { return $p.Value }
    return $null
}

function _Round4 {
    param($Value)
    if ($null -ne $Value) { return [Math]::Round([double]$Value, 4) }
    return $null
}

function _Pct {
    # Match Python's "{:.0%}" (e.g. 0.05 -> "5%"), no space before '%'.
    param($Value)
    return ('{0:0}%' -f ($Value * 100))
}

function Get-QualityReport {
    param($Segments, $Duration, [hashtable]$AudioStats, [hashtable]$Thresholds)
    $t = if ($Thresholds) { $Thresholds } else { Get-Thresholds }
    $reasons = New-Object System.Collections.Generic.List[string]
    $metrics = @{}
    $bad = $false; $suspect = $false

    if ($null -ne $Segments) {
        $segList = @($Segments)
        $text = (($segList | ForEach-Object { [string](_Prop $_ 'text') }) -join '').Trim()
        $speech = 0.0
        foreach ($s in $segList) {
            $speech += [Math]::Max(0.0, [double](_Prop $s 'end') - [double](_Prop $s 'start'))
        }
        $coverage = if ($Duration -and $Duration -gt 0) { $speech / $Duration } else { $null }
        $avgNoSpeech = _MeanOrNull ($segList | ForEach-Object { _Prop $_ 'no_speech_prob' })
        $avgLogprob = _MeanOrNull ($segList | ForEach-Object { _Prop $_ 'avg_logprob' })
        $comps = @($segList | ForEach-Object { _Prop $_ 'compression_ratio' } | Where-Object { $null -ne $_ })
        $maxComp = if ($comps.Count) { ($comps | Measure-Object -Maximum).Maximum } else { $null }

        $metrics.num_segments = $segList.Count
        $metrics.char_count = $text.Length
        $metrics.speech_coverage = _Round4 $coverage
        $metrics.avg_no_speech_prob = _Round4 $avgNoSpeech
        $metrics.avg_logprob = _Round4 $avgLogprob
        $metrics.max_compression_ratio = _Round4 $maxComp

        if (-not $text -or $segList.Count -eq 0) {
            $bad = $true; $reasons.Add('no speech detected (empty transcript)')
        } else {
            $strongNoSpeech = ($null -ne $avgNoSpeech) -and ($avgNoSpeech -ge $t.MaxNoSpeech)
            $almostNone = ($null -ne $coverage) -and ($coverage -lt 0.02)
            if ($almostNone -or ($strongNoSpeech -and ($null -ne $coverage) -and ($coverage -lt $t.MinSpeech))) {
                $part = if ($null -ne $coverage) { '(coverage ' + (_Pct $coverage) } else { '(unknown coverage' }
                $part += if ($null -ne $avgNoSpeech) { ', no-speech {0:N2})' -f $avgNoSpeech } else { ')' }
                $bad = $true; $reasons.Add('little or no speech ' + $part)
            } else {
                if (($null -ne $coverage) -and ($coverage -lt $t.MinSpeech)) {
                    $suspect = $true; $reasons.Add('low speech coverage (' + (_Pct $coverage) + ')')
                }
                if ($strongNoSpeech) {
                    $suspect = $true; $reasons.Add(('high no-speech probability ({0:N2})' -f $avgNoSpeech))
                }
            }
            if (($null -ne $avgLogprob) -and ($avgLogprob -lt $t.MinLogprob)) {
                $suspect = $true; $reasons.Add(('low transcription confidence (avg_logprob {0:N2})' -f $avgLogprob))
            }
            if (($null -ne $maxComp) -and ($maxComp -gt $t.MaxCompression)) {
                $suspect = $true; $reasons.Add(('repetitive output (compression ratio {0:N2}) - possible noise' -f $maxComp))
            }
        }
    }

    if ($AudioStats) {
        $meanDb = $AudioStats['mean_volume_db']
        $sil = $AudioStats['silence_ratio']
        $metrics.mean_volume_db = $meanDb
        $metrics.max_volume_db = $AudioStats['max_volume_db']
        $metrics.silence_ratio = _Round4 $sil
        if (($null -ne $meanDb) -and ($meanDb -le $t.NearSilentDb)) {
            $bad = $true; $reasons.Add(('near-silent audio (mean volume {0:N0} dB)' -f $meanDb))
        }
        if ($null -ne $sil) {
            if ($sil -ge 0.98) { $bad = $true; $reasons.Add('almost entirely silence (' + (_Pct $sil) + ')') }
            elseif ($sil -ge $t.MostlySilence) { $suspect = $true; $reasons.Add('mostly silence (' + (_Pct $sil) + ')') }
        }
    }

    $verdict = if ($bad) { 'bad' } elseif ($suspect) { 'suspect' } else { 'ok' }
    return @{ verdict = $verdict; reasons = $reasons.ToArray(); metrics = $metrics }
}

function Format-QualitySummary {
    param([hashtable]$Report)
    $head = "quality: $($Report.verdict.ToUpper())"
    if ($Report.reasons.Count -gt 0) { return "$head - $(($Report.reasons) -join '; ')" }
    return $head
}

# --------------------------------------------------------------------------- #
# Prerequisite checker (-Check) - parity with plaude_local/preflight.py
# --------------------------------------------------------------------------- #
function Invoke-Check {
    $rows = @()
    $rows += [pscustomobject]@{ Name = 'FFmpeg';              Ok = (Test-Command 'ffmpeg');   Req = $true;  Remedy = 'winget install ffmpeg  (or https://www.gyan.dev/ffmpeg/builds/)' }
    $rows += [pscustomobject]@{ Name = 'ffprobe';             Ok = (Test-Command 'ffprobe');  Req = $true;  Remedy = 'ships with FFmpeg; install FFmpeg' }
    $rows += [pscustomobject]@{ Name = 'whisper-ctranslate2'; Ok = (Test-Command 'whisper-ctranslate2'); Req = $true; Remedy = 'pip install whisper-ctranslate2' }
    $gpu = (Test-Command 'nvidia-smi')
    $rows += [pscustomobject]@{ Name = 'NVIDIA GPU (optional)'; Ok = $true; Req = $false; Remedy = $(if ($gpu) { '' } else { 'no nvidia-smi found; will run on CPU' }) }
    $rows += [pscustomobject]@{ Name = 'DeepFilterNet (optional)'; Ok = (Test-Command 'deepFilter'); Req = $false; Remedy = 'pip install deepfilternet  (for -Denoise deepfilter)' }
    $rows += [pscustomobject]@{ Name = 'Resemble-Enhance (optional)'; Ok = (Test-Command 'resemble-enhance'); Req = $false; Remedy = 'pip install resemble-enhance  (for -Enhance resemble)' }
    $summ = Get-SummBackend -OllamaUrl $Script:DefaultOllamaUrl -LlamacppUrl $Script:DefaultLlamacppUrl
    $rows += [pscustomobject]@{ Name = 'Local LLM server (optional)'; Ok = [bool]$summ; Req = $false; Remedy = 'start Ollama (https://ollama.com/download) or llama.cpp server (for -Summarize)' }

    Write-Host 'plaude-local (PowerShell) environment check'
    Write-Host ('=' * 43)
    $missingReq = $false
    foreach ($r in $rows) {
        $sym = if ($r.Ok) { 'OK  ' } elseif ($r.Req) { 'FAIL' } else { 'WARN' }
        Write-Host ("[{0}] {1}" -f $sym, $r.Name)
        if (-not $r.Ok -and $r.Remedy) { Write-Host ("        -> {0}" -f $r.Remedy) }
        if ($r.Req -and -not $r.Ok) { $missingReq = $true }
    }
    Write-Host ''
    if ($missingReq) {
        Write-Host 'Result: MISSING required prerequisites (see FAIL lines above).'
        return 1
    }
    Write-Host 'Result: all required prerequisites satisfied.'
    return 0
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
function Invoke-Main {
    if ($Version) { Write-Host "plaude-local (PowerShell) $($Script:ToolVersion)"; return 0 }
    if ($Check) { return (Invoke-Check) }

    Set-HfOffline -Enabled $Offline

    if (-not $InputFile) {
        Write-ErrLine 'error: an input audio file is required (or use -Check to verify your setup).'
        return 2
    }
    if (-not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
        Write-ErrLine "error: input file not found: $InputFile"
        return 2
    }
    # Thorough prerequisite gate: ensure ffmpeg + ffprobe, offering to install
    # them or accept a path, else abort.
    if (-not (Confirm-Ffmpeg)) { return 3 }

    # FFmpeg is the arbiter of decodability (parity with the Python version).
    if (Test-Command 'ffprobe') {
        $codec = Get-AudioCodec -Path $InputFile
        if (-not $codec) {
            Write-ErrLine "error: FFmpeg found no decodable audio stream in $InputFile."
            return 10
        }
        Write-Log "      input audio codec: $codec"
    }

    # Fast, model-free triage: report a verdict and exit without transcribing.
    if ($AssessOnly) {
        $stats = Get-AudioStats -Path $InputFile
        $report = Get-QualityReport -AudioStats $stats -Thresholds (Get-Thresholds)
        Write-ErrLine (Format-QualitySummary $report)
        if ($report.verdict -eq 'bad' -and $OnBad -eq 'fail') { return 12 }
        return 0
    }

    $token = $HfToken
    if (-not $token -and $env:HF_TOKEN) { $token = $env:HF_TOKEN }
    if ($Diarize -and -not $token) {
        Write-ErrLine 'error: -Diarize requires a Hugging Face token via -HfToken or $env:HF_TOKEN.'
        return 4
    }

    # Fail fast if summarization requested but no server reachable.
    if ($Summarize) {
        $oUrl = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultOllamaUrl }
        $lUrl = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultLlamacppUrl }
        $reachable = switch ($SummarizeBackend) {
            'ollama'   { Test-OllamaUp $oUrl }
            'llamacpp' { Test-LlamacppUp $lUrl }
            default    { [bool](Get-SummBackend -OllamaUrl $oUrl -LlamacppUrl $lUrl) }
        }
        if (-not $reachable) {
            Write-ErrLine 'error: -Summarize needs a local LLM server (Ollama or llama.cpp). Start one and retry.'
            return 8
        }
    }

    # -ModelDir: whisper-ctranslate2 has no download-cache flag; redirect the
    # Hugging Face cache instead (closest equivalent to faster-whisper's
    # download_root). $env change is process-local.
    if ($ModelDir) { $env:HF_HOME = $ModelDir }

    # num/min/max speakers aren't supported by whisper-ctranslate2's diarization.
    if ($Diarize -and ($NumSpeakers -or $MinSpeakers -or $MaxSpeakers)) {
        Write-Log 'note: -NumSpeakers/-MinSpeakers/-MaxSpeakers are not supported by the PowerShell diarization engine; ignoring.'
    }

    $work = Join-Path ([System.IO.Path]::GetTempPath()) ("plaude-local-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    try {
        Write-Log "[1/4] preparing audio (denoise=$Denoise, enhance=$Enhance, gain=$(Format-Db $Gain)dB) ..."
        try {
            $prepared = Invoke-Prepare -Src $InputFile -WorkDir $work
        } catch { Write-ErrLine "error: $($_.Exception.Message)"; return 5 }
        if ($KeepClean) {
            Copy-Item -LiteralPath $prepared -Destination $KeepClean -Force
            Write-Log "      saved cleaned audio -> $KeepClean"
        }

        $dev = Resolve-Device
        $compute = Resolve-ComputeType -Dev $dev
        $diarNote = if ($Diarize) { ' +diarize' } else { '' }
        Write-Log "[2/4] transcribing (model=$Model device=$dev compute=$compute$diarNote) ..."
        try {
            $producedPath = Invoke-Transcribe -AudioPath $prepared -WorkDir $work -Dev $dev -Compute $compute -Token $token
        } catch { Write-ErrLine "error: $($_.Exception.Message)"; return 6 }
        $transcriptText = [System.IO.File]::ReadAllText($producedPath, [System.Text.Encoding]::UTF8)

        Write-Log $(if ($Diarize) { '[3/4] diarization handled by the engine' } else { '[3/4] diarization skipped' })

        # Assess recording quality from the JSON metrics whisper-ctranslate2 wrote.
        $jsonPath = Join-Path $work ([System.IO.Path]::GetFileNameWithoutExtension($prepared) + '.json')
        $report = $null
        if (Test-Path -LiteralPath $jsonPath) {
            $jsonObj = (Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8) | ConvertFrom-Json
            $segs = if ($jsonObj.PSObject.Properties['segments']) { $jsonObj.segments } else { @() }
            $report = Get-QualityReport -Segments $segs -Duration (Get-MediaDuration $InputFile) -Thresholds (Get-Thresholds)
            Write-Log ('      ' + (Format-QualitySummary $report))
        }

        # A bad recording triggers the -OnBad policy (default: warn and continue).
        if ($report -and $report.verdict -eq 'bad') {
            if ($OnBad -eq 'skip') { Write-ErrLine ('skipped: bad recording - ' + (Format-QualitySummary $report)); return 0 }
            if ($OnBad -eq 'fail') { Write-ErrLine ('error: bad recording - ' + (Format-QualitySummary $report)); return 12 }
            Write-ErrLine ('warning: ' + (Format-QualitySummary $report) + ' (writing anyway; use -OnBad to change)')
        }

        # For JSON output, surface the quality report (top-level field; the
        # Python port nests it under meta.quality - see Known differences).
        if ($Format -eq 'json' -and $report) {
            try {
                $obj = $transcriptText | ConvertFrom-Json
                $obj | Add-Member -NotePropertyName quality -NotePropertyValue $report -Force
                $transcriptText = $obj | ConvertTo-Json -Depth 12
            } catch { }
        }

        # 4. Write transcript (UTF-8, output.<fmt> default, overwrite guard)
        $transcriptOut = if ($Output) { $Output } else { Get-DefaultOutput -Fmt $Format }
        if ($transcriptOut -eq '-') {
            [Console]::Out.Write($transcriptText)
        } else {
            if (-not (Confirm-Overwrite -Path $transcriptOut)) {
                Write-ErrLine "aborted: $transcriptOut was not overwritten."
                return 11
            }
            try {
                Write-Utf8File -Path $transcriptOut -Text $transcriptText
            } catch { Write-ErrLine "error: could not write transcript to ${transcriptOut}: $($_.Exception.Message)"; return 9 }
            Write-Log "[4/4] transcript -> $transcriptOut"
        }

        # 5. Summarize (optional) - always summarize a PLAIN-TEXT transcript.
        if ($Summarize) {
            Write-Log "      summarizing (backend=$SummarizeBackend) ..."
            $plainPath = Join-Path $work ([System.IO.Path]::GetFileNameWithoutExtension($prepared) + '.txt')
            $plainText = if (Test-Path -LiteralPath $plainPath) {
                [System.IO.File]::ReadAllText($plainPath, [System.Text.Encoding]::UTF8)
            } else { $transcriptText }

            $backend = $SummarizeBackend
            if ($backend -eq 'auto') {
                $oUrl = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultOllamaUrl }
                $lUrl = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultLlamacppUrl }
                $backend = Get-SummBackend -OllamaUrl $oUrl -LlamacppUrl $lUrl
            }
            $url = if ($SummarizeUrl) { $SummarizeUrl }
                   elseif ($backend -eq 'ollama') { $Script:DefaultOllamaUrl }
                   else { $Script:DefaultLlamacppUrl }
            $model = if ($SummarizeModel) { $SummarizeModel } else { $Script:DefaultOllamaModel }

            try {
                $summary = Invoke-Summarize -Text $plainText -Backend $backend -Model $model -Url $url -MaxChars $SummarizeMaxChars
            } catch { Write-ErrLine "error: $($_.Exception.Message)"; return 8 }
            $body = "# Summary`n`n" + $summary.Trim() + "`n"

            $toStdout = ($SummaryOutput -eq '-') -or ($transcriptOut -eq '-' -and -not $SummaryOutput)
            if ($toStdout) {
                [Console]::Out.Write("`n" + $body)
            } else {
                $summaryOut = if ($SummaryOutput) { $SummaryOutput }
                              else { [System.IO.Path]::ChangeExtension($transcriptOut, '.summary.md') }
                try {
                    Write-Utf8File -Path $summaryOut -Text $body
                } catch { Write-ErrLine "error: could not write summary to ${summaryOut}: $($_.Exception.Message)"; return 9 }
                Write-Log "      summary -> $summaryOut"
            }
        }
        return 0
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Only run when invoked directly - NOT when dot-sourced (e.g. by the Pester
# tests), so the functions above can be tested in isolation.
if ($MyInvocation.InvocationName -ne '.') {
    try {
        exit (Invoke-Main)
    } catch {
        Write-ErrLine "error: $($_.Exception.Message)"
        exit 1
    }
}
