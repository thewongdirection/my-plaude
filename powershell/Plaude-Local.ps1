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

    [ValidateSet('txt', 'srt', 'vtt', 'json', 'html')]
    [string]$Format = 'html',

    [string]$TranslateTo = 'en',
    [string]$TranslateModel,
    [ValidateSet('auto', 'whisper', 'llm')]
    [string]$TranslateEngine = 'auto',
    [switch]$SplitOutputs,
    [string]$TranscriptionFile,
    [string]$TranslationFile,

    [Alias('Doctor')]
    [switch]$Check,
    [switch]$ListModels,

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
    [double]$SummarizeTimeout = 120,

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
# Use 127.0.0.1, not 'localhost': Invoke-RestMethod resolves 'localhost' to IPv6
# ::1 first and stalls for seconds before falling back to IPv4, which blew past
# the 2s backend-detection timeout and made a running Ollama look unreachable
# (translation/summary silently degraded to "unavailable"). 127.0.0.1 is instant.
$Script:DefaultOllamaUrl = 'http://127.0.0.1:11434'
$Script:DefaultLlamacppUrl = 'http://127.0.0.1:8080'
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
    param([double]$TimeoutSeconds)
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
function Disable-HfTelemetry {
    # Local-first: no Hugging Face telemetry on any run. Set-default style (an
    # explicit opt-in is respected). Parity with Python cli.disable_telemetry().
    if (-not $env:HF_HUB_DISABLE_TELEMETRY) { $env:HF_HUB_DISABLE_TELEMETRY = '1' }
    if (-not $env:DISABLE_TELEMETRY) { $env:DISABLE_TELEMETRY = '1' }
}

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
    param([string]$AudioPath, [string]$WorkDir, [string]$Dev, [string]$Compute,
          [string]$Token, [string]$Task = 'transcribe')

    # Always request 'all' so we get JSON (per-segment quality metrics) and a
    # plain-text copy (for summarization) alongside the requested format.
    $a = @(
        $AudioPath,
        '--model', $Model,
        '--device', $Dev,
        '--compute_type', $Compute,
        '--task', $Task,
        '--output_dir', $WorkDir,
        '--output_format', 'all',
        '--beam_size', "$BeamSize",
        # Always pass VAD explicitly (parity with Python's vad_filter=not no_vad).
        '--vad_filter', $(if ($NoVad) { 'False' } else { 'True' })
    )
    if ($Language) { $a += @('--language', $Language) }
    # NOTE: diarization is NOT delegated to whisper-ctranslate2 (--hf_token). Its
    # built-in diarization calls pyannote's 4.x `token=` API, which can't run on
    # Windows (pyannote 4.x needs k2, no Windows wheels). We diarize separately via
    # Invoke-PyannoteDiarize + Merge-Turns (pyannote 3.x/4.x compatible), matching
    # the Python implementation and honoring -Num/-Min/-MaxSpeakers/-DiarizeModel.

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
    # whisper-ctranslate2 does not emit 'html'; the dashboard reads the txt copy.
    $ext = if ($Format -eq 'html') { 'txt' } else { $Format }
    $produced = Join-Path $WorkDir "$base.$ext"
    if (-not (Test-Path -LiteralPath $produced)) {
        throw "expected transcript not found at $produced"
    }
    return $produced
}

# --------------------------------------------------------------------------- #
# Speaker diarization - parity with plaude_local/diarize.py.
# Runs pyannote directly (via pyannote_diarize.py) instead of whisper-ctranslate2's
# built-in diarization, so it works with the Windows-viable pyannote 3.1.x, and
# merges the turns onto the transcript segments here (Merge-Turns).
# --------------------------------------------------------------------------- #
function Invoke-PyannoteDiarize {
    # Returns turns as an array of [start, end, label] (empty array on none).
    param([string]$AudioPath, [string]$Token, [string]$Device, [string]$Model,
          [int]$Num, [int]$Min, [int]$Max)
    $py = Resolve-PythonExe
    if (-not $py) { throw 'python (or py) must be on PATH to run diarization (pyannote).' }
    $helper = Join-Path $PSScriptRoot 'pyannote_diarize.py'
    if (-not (Test-Path -LiteralPath $helper)) { throw "diarization helper not found at $helper" }
    $a = @($helper, $AudioPath, '--device', $Device)
    if ($Token) { $a += @('--token', $Token) }
    if ($Model) { $a += @('--model', $Model) }
    if ($Num -gt 0) { $a += @('--num-speakers', "$Num") }
    if ($Min -gt 0) { $a += @('--min-speakers', "$Min") }
    if ($Max -gt 0) { $a += @('--max-speakers', "$Max") }
    # stdout carries the JSON turns; capture stderr (progress/errors) to a
    # UNIQUE temp file (overwrite, not append) so a failure reports only this
    # run's message, then always clean it up.
    $errFile = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(),
        ('plaude-diar-' + [Guid]::NewGuid().ToString('N') + '.err'))
    try {
        $out = & $py @a 2> $errFile
        if ($LASTEXITCODE -ne 0) {
            $errTxt = if (Test-Path -LiteralPath $errFile) { (Get-Content -LiteralPath $errFile -Raw) } else { '' }
            throw "diarization failed: $($errTxt.Trim())"
        }
    } finally {
        if (Test-Path -LiteralPath $errFile) { Remove-Item -LiteralPath $errFile -ErrorAction SilentlyContinue }
    }
    $json = ($out | Out-String).Trim()
    if (-not $json) { return @() }
    # Convert the positional [start,end,label] JSON turns into objects so a SINGLE
    # turn doesn't get unwrapped into three scalars (PowerShell array semantics).
    $turns = New-Object System.Collections.Generic.List[object]
    foreach ($t in @($json | ConvertFrom-Json)) {
        $turns.Add([pscustomobject]@{ Start = [double]$t[0]; End = [double]$t[1]; Label = [string]$t[2] })
    }
    return , $turns.ToArray()
}

function Get-SpeakerMap {
    # Map raw backend labels (SPEAKER_00...) to "Speaker N", first-seen by start
    # time (parity with diarize._relabel). $Turns = objects {Start,End,Label}.
    param($Turns)
    $map = [ordered]@{}
    foreach ($t in (@($Turns) | Sort-Object { [double]$_.Start })) {
        $label = [string]$t.Label
        if (-not $map.Contains($label)) { $map[$label] = "Speaker $($map.Count + 1)" }
    }
    return $map
}

function Get-BestSpeaker {
    # Speaker whose turn overlaps [Start,End] the most (parity with _best_speaker).
    param([double]$Start, [double]$End, $Turns)
    $best = $null; $bestOverlap = 0.0
    foreach ($t in @($Turns)) {
        $overlap = [Math]::Min($End, [double]$t.End) - [Math]::Max($Start, [double]$t.Start)
        if ($overlap -gt $bestOverlap) { $bestOverlap = $overlap; $best = [string]$t.Label }
    }
    return $best
}

function Merge-Turns {
    # Tag each segment with its best-overlap speaker (parity with merge_turns; the
    # no-words branch, since whisper-ctranslate2 segments carry no word timings).
    param($Segments, $Turns)
    $segList = @($Segments)
    if (-not $Turns -or @($Turns).Count -eq 0) { return $segList }
    $map = Get-SpeakerMap -Turns $Turns
    foreach ($s in $segList) {
        $label = Get-BestSpeaker -Start ([double]$s.start) -End ([double]$s.end) -Turns $Turns
        $name = if ($label -and $map.Contains($label)) { $map[$label] } else { $null }
        if ($s.PSObject.Properties['speaker']) { $s.speaker = $name }
        else { $s | Add-Member -NotePropertyName speaker -NotePropertyValue $name -Force }
    }
    return $segList
}

function Format-DiarizedTranscript {
    # Rebuild the transcript from speaker-tagged segments for the requested format
    # (parity with plaude_local.formats: to_text/to_srt/to_vtt/to_json + speaker).
    param($Segments, [string]$Fmt)
    $segList = @($Segments)
    if ($Fmt -eq 'srt') {
        $blocks = New-Object System.Collections.Generic.List[string]; $i = 1
        foreach ($s in $segList) {
            $text = ([string]$s.text).Trim(); if (-not $text) { continue }
            if ($s.speaker) { $text = "$($s.speaker): $text" }
            $blocks.Add("$i`n$(Format-SrtTime ([double]$s.start)) --> $(Format-SrtTime ([double]$s.end))`n$text`n"); $i++
        }
        return ($blocks -join "`n")
    }
    if ($Fmt -eq 'vtt') {
        $blocks = New-Object System.Collections.Generic.List[string]; $blocks.Add("WEBVTT`n")
        foreach ($s in $segList) {
            $text = ([string]$s.text).Trim(); if (-not $text) { continue }
            if ($s.speaker) { $text = "<v $($s.speaker)>$text" }
            $blocks.Add("$(Format-VttTime ([double]$s.start)) --> $(Format-VttTime ([double]$s.end))`n$text`n")
        }
        return ($blocks -join "`n")
    }
    if ($Fmt -eq 'json') {
        # Force a JSON array for the segment list even when there is one segment
        # (Windows PowerShell 5.1's ConvertTo-Json unwraps a single-element array).
        $segJson = @($segList | ForEach-Object { $_ | ConvertTo-Json -Depth 12 -Compress }) -join ','
        return "{`n  `"segments`": [$segJson]`n}"
    }
    # txt / html: turn-by-turn lines. The 'txt' format shows a [MM:SS] clock
    # (parity with meta.timestamps=diarize -> to_text); the html dashboard does
    # not (to_text(segments) with timestamps=False).
    $withClock = ($Fmt -eq 'txt')
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($s in $segList) {
        $text = ([string]$s.text).Trim(); if (-not $text) { continue }
        $parts = New-Object System.Collections.Generic.List[string]
        if ($withClock) { $parts.Add("[$(Format-Clock ([double]$s.start))]") }
        if ($s.speaker) { $parts.Add("$($s.speaker):") }
        $prefix = ($parts -join ' ')
        $lines.Add($(if ($prefix) { "$prefix $text" } else { $text }))
    }
    if ($lines.Count -eq 0) { return '' }
    return (($lines -join "`n") + "`n")
}

function Format-Clock {
    # Short [MM:SS] / [H:MM:SS] clock for plain text (parity with formats._clock).
    param([double]$Seconds)
    if ($Seconds -lt 0) { $Seconds = 0.0 }
    $total = [int][Math]::Round($Seconds)
    $h = [Math]::Floor($total / 3600); $rem = $total - $h * 3600
    $m = [Math]::Floor($rem / 60); $s = $rem - $m * 60
    if ($h -gt 0) { return ('{0:0}:{1:00}:{2:00}' -f $h, $m, $s) }
    return ('{0:00}:{1:00}' -f $m, $s)
}

function Format-SrtTime {
    param([double]$Seconds)
    if ($Seconds -lt 0) { $Seconds = 0.0 }
    $ms = [int][Math]::Round($Seconds * 1000.0)
    $h = [Math]::Floor($ms / 3600000); $ms -= $h * 3600000
    $m = [Math]::Floor($ms / 60000); $ms -= $m * 60000
    $s = [Math]::Floor($ms / 1000); $ms -= $s * 1000
    return ('{0:00}:{1:00}:{2:00},{3:000}' -f $h, $m, $s, $ms)
}

function Format-VttTime {
    param([double]$Seconds)
    return (Format-SrtTime $Seconds).Replace(',', '.')
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
    param([string]$Prompt, [string]$Backend, [string]$Model, [string]$Url, [double]$TimeoutSec = 120)
    # Encode the JSON body as UTF-8 bytes so non-Latin (CJK) transcript text is
    # sent correctly regardless of the default request encoding.
    if ($Backend -eq 'ollama') {
        # Fixed context window (parity with _OLLAMA_NUM_CTX). num_ctx is a
        # load-time parameter, so a value that varies per call forces Ollama to
        # reload the model every request (timeouts/500s). One constant keeps it
        # warm while still beating the truncating ~4k default.
        $json = @{ model = $Model; prompt = $Prompt; stream = $false; options = @{ num_ctx = $Script:OllamaNumCtx } } | ConvertTo-Json -Depth 4
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
        $resp = Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/api/generate" -Method Post `
            -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec $TimeoutSec
        if (-not $resp.response) { throw 'ollama returned no text.' }
        return ([string]$resp.response).Trim()
    } else {
        $json = @{ prompt = $Prompt; n_predict = 512; temperature = 0.2; stream = $false } | ConvertTo-Json
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
        $resp = Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/completion" -Method Post `
            -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec $TimeoutSec
        if (-not $resp.content) { throw 'llama.cpp returned no text.' }
        return ([string]$resp.content).Trim()
    }
}

function Invoke-Summarize {
    param(
        [string]$Text, [string]$Backend, [string]$Model, [string]$Url, [int]$MaxChars,
        [int]$Depth = 0, [switch]$Topics, [double]$TimeoutSec = 120
    )
    $Text = $Text.Trim()
    if (-not $Text) { throw 'nothing to summarize: the transcript is empty.' }
    $MaxChars = [Math]::Max($MaxChars, 1)
    # Select the prompt builder: dashboard 'topics' (<=250 words) or bullets.
    $pf = {
        param([string]$t, [switch]$c)
        if ($Topics) { Get-CriticalTopicsPrompt -Text $t -Combine:$c }
        else { Build-SummaryPrompt -Text $t -Combine:$c }
    }

    $chunks = Split-IntoChunks -Text $Text -MaxChars $MaxChars
    if ($chunks.Count -le 1) {
        return Invoke-LlmCall -Prompt (& $pf $chunks[0]) -Backend $Backend -Model $Model -Url $Url -TimeoutSec $TimeoutSec
    }
    $partials = foreach ($c in $chunks) {
        Invoke-LlmCall -Prompt (& $pf $c) -Backend $Backend -Model $Model -Url $Url -TimeoutSec $TimeoutSec
    }
    $combined = ($partials -join "`n`n")
    # Map-reduce (parity with Python's summarize_text): if the combined partial
    # summaries still exceed the budget, reduce again - bounded to depth 3 so it
    # always terminates.
    if ($combined.Length -le $MaxChars -or $Depth -ge 3) {
        return Invoke-LlmCall -Prompt (& $pf $combined -c) -Backend $Backend -Model $Model -Url $Url -TimeoutSec $TimeoutSec
    }
    return Invoke-Summarize -Text $combined -Backend $Backend -Model $Model -Url $Url -MaxChars $MaxChars -Depth ($Depth + 1) -Topics:$Topics -TimeoutSec $TimeoutSec
}

function Build-TranslatePrompt {
    param([string]$Text, [string]$TargetLanguage)
    return "Translate the following text into $TargetLanguage. Output ONLY the translation, preserving line breaks and meaning, with no preamble, notes, or the original text.`n`nText:`n$Text`n`n$TargetLanguage translation:"
}

function Invoke-TranslateText {
    param([string]$Text, [string]$TargetLanguage, [string]$Backend, [string]$Model, [string]$Url, [int]$MaxChars, [double]$TimeoutSec = 120)
    $Text = $Text.Trim()
    if (-not $Text) { return '' }
    $chunks = Split-IntoChunks -Text $Text -MaxChars $MaxChars
    $parts = foreach ($c in $chunks) {
        Invoke-LlmCall -Prompt (Build-TranslatePrompt -Text $c -TargetLanguage $TargetLanguage) -Backend $Backend -Model $Model -Url $Url -TimeoutSec $TimeoutSec
    }
    return ($parts -join "`n")
}

function Get-OllamaModels {
    # Names of every model installed on the Ollama server (empty if unreachable).
    param([string]$Url = $Script:DefaultOllamaUrl)
    try {
        $resp = Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/api/tags" -TimeoutSec 3
        if ($resp.models) { return @($resp.models | ForEach-Object { [string]$_.name } | Where-Object { $_ }) }
    } catch { }
    return @()
}

function Get-DefaultOllamaModel {
    # The model to use when none is specified: the first Ollama has installed.
    param([string]$Url = $Script:DefaultOllamaUrl)
    $models = @(Get-OllamaModels -Url $Url)
    if ($models.Count -gt 0) { return $models[0] }
    return $null
}

function Build-AlignedTranslatePrompt {
    param([string]$Numbered, [string]$TargetLanguage)
    return "Translate each numbered line below into $TargetLanguage. Output EXACTLY the same number of lines, each starting with its number and a period, then the $TargetLanguage translation of that line only. Consider the whole passage for context, but do not merge, split, reorder, add, or drop lines, and output nothing but the numbered $TargetLanguage lines.`n`n$Numbered`n`n$TargetLanguage (numbered):"
}

# Cap lines per translation request (parity with DEFAULT_MAX_TRANSLATE_LINES):
# very large batches overflow the LLM context and the numbered reply parses to
# nothing. Small batches translate reliably and align cleanly.
$Script:MaxTranslateLines = 40
# Fixed Ollama context window (parity with _OLLAMA_NUM_CTX): constant so the
# model stays warm, 2x the ~4k default so batches/chunks aren't truncated.
$Script:OllamaNumCtx = 8192

function Invoke-TranslateGroup {
    # Translate one group of line indices into $Out (a string[] mutated in place).
    # If the reply maps back to fewer than half the lines (context overflow ->
    # truncated/misformatted), split the group and retry each half, down to
    # single lines, so a batch never silently comes back all-empty.
    param([int[]]$Group, [string[]]$Lines, [string]$TargetLanguage, [string]$Backend, [string]$Model, [string]$Url, [double]$TimeoutSec, [string[]]$Out)
    $numberedLines = for ($k = 0; $k -lt $Group.Count; $k++) { "$($k + 1). $($Lines[$Group[$k]])" }
    $numbered = $numberedLines -join "`n"
    $resp = Invoke-LlmCall -Prompt (Build-AlignedTranslatePrompt -Numbered $numbered -TargetLanguage $TargetLanguage) -Backend $Backend -Model $Model -Url $Url -TimeoutSec $TimeoutSec
    $resp = [regex]::Replace($resp, '(?is)<think>.*?</think>', '')
    $got = @{}
    foreach ($m in [regex]::Matches($resp, '(?m)^\s*(\d+)[.\):]\s*(.*)$')) { $got[[int]$m.Groups[1].Value] = $m.Groups[2].Value.Trim() }
    $matched = 0
    for ($n = 0; $n -lt $Group.Count; $n++) { if ($got.ContainsKey($n + 1) -and $got[$n + 1]) { $matched++ } }
    if ($Group.Count -eq 1 -or ($matched * 2) -ge $Group.Count) {
        for ($n = 0; $n -lt $Group.Count; $n++) {
            $Out[$Group[$n]] = if ($got.ContainsKey($n + 1)) { $got[$n + 1] } else { '' }
        }
        return
    }
    $mid = [int][Math]::Floor($Group.Count / 2)
    Invoke-TranslateGroup -Group $Group[0..($mid - 1)] -Lines $Lines -TargetLanguage $TargetLanguage -Backend $Backend -Model $Model -Url $Url -TimeoutSec $TimeoutSec -Out $Out
    Invoke-TranslateGroup -Group $Group[$mid..($Group.Count - 1)] -Lines $Lines -TargetLanguage $TargetLanguage -Backend $Backend -Model $Model -Url $Url -TimeoutSec $TimeoutSec -Out $Out
}

function Invoke-TranslateLines {
    # Accuracy-first, alignment-preserving LLM translation: one output line per
    # input segment line. Parity with summarize.translate_lines.
    param([string[]]$Lines, [string]$TargetLanguage, [string]$Backend, [string]$Model, [string]$Url, [int]$MaxChars, [double]$TimeoutSec = 120, [int]$MaxLines = 0)
    if ($MaxLines -le 0) { $MaxLines = $Script:MaxTranslateLines }
    $out = New-Object 'string[]' $Lines.Count
    for ($i = 0; $i -lt $out.Count; $i++) { $out[$i] = '' }
    $batches = New-Object System.Collections.Generic.List[object]
    $cur = New-Object System.Collections.Generic.List[int]; $clen = 0
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        $add = $Lines[$i].Length + 6
        if ($cur.Count -gt 0 -and (($clen + $add) -gt $MaxChars -or $cur.Count -ge $MaxLines)) {
            $batches.Add($cur.ToArray()); $cur = New-Object System.Collections.Generic.List[int]; $clen = 0
        }
        $cur.Add($i); $clen += $add
    }
    if ($cur.Count -gt 0) { $batches.Add($cur.ToArray()) }
    foreach ($group in $batches) {
        Invoke-TranslateGroup -Group $group -Lines $Lines -TargetLanguage $TargetLanguage -Backend $Backend -Model $Model -Url $Url -TimeoutSec $TimeoutSec -Out $out
    }
    return , $out
}

function Resolve-TranslateEngine {
    if ($TranslateEngine -ne 'auto') { return $TranslateEngine }
    $oUrl = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultOllamaUrl }
    $lUrl = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultLlamacppUrl }
    if (Get-SummBackend -OllamaUrl $oUrl -LlamacppUrl $lUrl) { return 'llm' } else { return 'whisper' }
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

function Invoke-ListModels {
    # List installed Ollama models, marking the translation default; exit.
    # Parity with plaude_local.cli._run_list_models.
    $url = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultOllamaUrl }
    $models = @(Get-OllamaModels -Url $url)
    if ($models.Count -eq 0) {
        Write-Host "No local LLM models found (is Ollama running at ${url}?)."
        Write-Host 'Install Ollama (https://ollama.com) and pull one, e.g.:'
        Write-Host '  ollama pull qwen2.5:7b'
        return 1
    }
    $default = if ($TranslateModel) { $TranslateModel } else { $models[0] }
    Write-Host "Local LLM models on Ollama ($url):"
    foreach ($m in $models) {
        if ($m -eq $default) { Write-Host "  * $m   <- default for translation" }
        else { Write-Host "    $m" }
    }
    Write-Host ''
    if ($TranslateModel -and ($models -notcontains $TranslateModel)) {
        Write-Host "note: -TranslateModel '$TranslateModel' is not installed; pull it with ``ollama pull $TranslateModel``."
    }
    Write-Host 'Choose a model with -TranslateModel NAME (translation) or -SummarizeModel NAME (summary).'
    return 0
}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# HTML dashboard (parity with plaude_local/dashboard.py)
# --------------------------------------------------------------------------- #
$Script:LangNames = @{
    en = 'English'; zh = 'Chinese'; ms = 'Malay'; es = 'Spanish'; fr = 'French'
    de = 'German'; ja = 'Japanese'; ko = 'Korean'; id = 'Indonesian'; hi = 'Hindi'
    ar = 'Arabic'; pt = 'Portuguese'; ru = 'Russian'; it = 'Italian'; vi = 'Vietnamese'
    th = 'Thai'; nl = 'Dutch'; tr = 'Turkish'; pl = 'Polish'; uk = 'Ukrainian'
}
$Script:CjkRegex = ('[{0}-{1}{2}-{3}{4}-{5}{6}-{7}]' -f [char]0x3400,[char]0x9fff,[char]0xf900,[char]0xfaff,[char]0x3040,[char]0x30ff,[char]0xac00,[char]0xd7af)

function Get-LanguageName {
    param([string]$Code)
    if (-not $Code) { return 'Unknown' }
    $c = $Code.ToLower()
    if ($Script:LangNames.ContainsKey($c)) { return $Script:LangNames[$c] }
    return $Code
}

function Get-WordCount {
    param([string]$Text)
    if (-not $Text) { return 0 }
    $cjk = ([regex]::Matches($Text, $Script:CjkRegex)).Count
    $spaced = @([regex]::Replace($Text, $Script:CjkRegex, ' ') -split '\s+' |
        Where-Object { $_ -ne '' }).Count
    return $cjk + $spaced
}

function Format-SpeechDuration {
    param($Seconds)
    if ($null -eq $Seconds) { return [char]0x2014 }  # em dash
    $s = [int][Math]::Round([double]$Seconds)
    $h = [int][Math]::Floor($s / 3600); $m = [int][Math]::Floor(($s % 3600) / 60); $sec = [int]($s % 60)
    if ($h -gt 0) { return ('{0}h {1:d2}m {2:d2}s' -f $h, $m, $sec) }
    if ($m -gt 0) { return ('{0}m {1:d2}s' -f $m, $sec) }
    return "${sec}s"
}

function Get-CriticalTopicsPrompt {
    param([string]$Text, [switch]$Combine, [int]$MaxWords = 250)
    if ($Combine) {
        $head = "Below are partial notes from a longer transcript. Combine them into a single summary of the most critical topics, in at most $MaxWords words. No preamble. Preserve the original language of the transcript."
        $label = 'Partial notes'
    } else {
        $head = "Summarize the most critical topics discussed in the following transcript in at most $MaxWords words. Focus on what matters most; no preamble or meta commentary. Preserve the original language."
        $label = 'Transcript'
    }
    return "$head`n`n${label}:`n$Text`n`nSummary:"
}

function Get-AlignedPairs {
    # Pair original segments with time-overlapping translation text (parity with
    # dashboard.align_segments). Returns [pscustomobject]@{ O=orig; X=trans } rows.
    param($OrigSegments, $TransSegments)
    $trans = @()
    foreach ($t in @($TransSegments)) {
        if ($null -ne $t) { $trans += , @([double]$t.start, [double]$t.end, ([string]$t.text).Trim()) }
    }
    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($o in @($OrigSegments)) {
        if ($null -eq $o) { continue }
        $os = [double]$o.start; $oe = [double]$o.end
        $matched = foreach ($tr in $trans) {
            if ($tr[2] -and ($os -le (($tr[0] + $tr[1]) / 2.0)) -and ((($tr[0] + $tr[1]) / 2.0) -lt $oe)) { $tr[2] }
        }
        $rows.Add([pscustomobject]@{ O = ([string]$o.text).Trim(); X = (($matched) -join ' ') })
    }
    return $rows.ToArray()
}

function Get-DashboardHtml {
    param(
        [string]$Title, [string]$Language, $SpeechDurationS, [int]$WordCount,
        [string]$Summary, [string]$SummaryNote, [string]$Transcript,
        [string]$Translation, [string]$TranslationLabel, [bool]$Cjk, $Pairs,
        [string]$TranscriptionEngine = '', [string]$TranslationEngine = ''
    )
    $css = @'
:root{--bg:#f6f7f9;--surface:#fff;--text:#1a1d23;--muted:#5b6270;--accent:#0e7c86;--accent-ink:#0a5b62;--accent-soft:#e2f1f1;--border:#e4e7eb;--shadow:0 1px 2px rgba(20,25,35,.04),0 8px 24px rgba(20,25,35,.06);--font-sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;--font-read:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Songti SC","Noto Serif CJK SC","Microsoft YaHei",serif;--font-mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--surface:#171b21;--text:#e8eaed;--muted:#9aa2ad;--accent:#43c9c0;--accent-ink:#7ee0d8;--accent-soft:#12332f;--border:#262b33;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}
:root[data-theme="light"]{--bg:#f6f7f9;--surface:#fff;--text:#1a1d23;--muted:#5b6270;--accent:#0e7c86;--accent-ink:#0a5b62;--accent-soft:#e2f1f1;--border:#e4e7eb;}
:root[data-theme="dark"]{--bg:#0f1216;--surface:#171b21;--text:#e8eaed;--muted:#9aa2ad;--accent:#43c9c0;--accent-ink:#7ee0d8;--accent-soft:#12332f;--border:#262b33;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--font-sans);line-height:1.5;-webkit-font-smoothing:antialiased;}
.wrap{max-width:860px;margin:0 auto;padding:clamp(20px,4vw,56px) clamp(16px,4vw,28px) 72px;}
.kicker{font-family:var(--font-mono);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 12px;}
h1{font-family:var(--font-read);font-weight:600;font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.1;margin:0 0 20px;}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 26px;}
.stat{flex:1 1 150px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px;box-shadow:var(--shadow);}
.stat .label{font-family:var(--font-mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 6px;}
.stat .value{font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1;}
.prov{display:flex;gap:20px;flex-wrap:wrap;font-family:var(--font-mono);font-size:.72rem;color:var(--muted);margin:0 0 24px;}
.prov b{color:var(--text);font-weight:600;}
.summary{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:12px;padding:18px 22px;box-shadow:var(--shadow);margin:0 0 8px;}
.summary h2{font-family:var(--font-read);font-size:1.15rem;margin:0 0 10px;}
.summary .note{color:var(--muted);font-style:italic;}
.tabs{display:flex;gap:4px;margin:28px 0 0;border-bottom:1px solid var(--border);flex-wrap:wrap;}
.tab{appearance:none;border:0;background:transparent;cursor:pointer;font-family:var(--font-sans);font-size:.94rem;color:var(--muted);padding:11px 15px;border-bottom:2px solid transparent;margin-bottom:-1px;}
.tab[aria-selected="true"]{color:var(--text);border-bottom-color:var(--accent);font-weight:600;}
.panel{margin-top:22px;}
.panel[hidden]{display:none;}
.doc{background:var(--surface);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);padding:clamp(20px,3.5vw,34px);overflow-x:auto;}
.doc pre{margin:0;font-family:var(--font-read);font-size:1.03rem;line-height:1.72;white-space:pre-wrap;word-break:break-word;}
.doc.cjk pre{line-height:1.95;font-size:1.08rem;}
.sbs{display:grid;grid-template-columns:1fr 1fr;background:var(--surface);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);overflow:hidden;}
.sbs-h{position:sticky;top:0;background:var(--surface);font-family:var(--font-mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);padding:12px 18px;border-bottom:1px solid var(--border);z-index:1;}
.sbs-o,.sbs-x{padding:11px 18px;border-bottom:1px solid var(--border);font-family:var(--font-read);font-size:1.01rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;}
.sbs-o{border-right:1px solid var(--border);}
.sbs-o.cjk{line-height:1.9;font-size:1.06rem;}
.sbs-x.empty{color:var(--muted);font-style:italic;}
footer{margin-top:36px;padding-top:18px;border-top:1px solid var(--border);color:var(--muted);font-size:.8rem;font-family:var(--font-mono);}
'@
    $js = @'
const IDS=["transcribed","translation","sbs"];
function sel(id){IDS.forEach(x=>{const on=x===id;document.getElementById("tab-"+x).setAttribute("aria-selected",on);document.getElementById("panel-"+x).hidden=!on;});}
IDS.forEach(x=>document.getElementById("tab-"+x).addEventListener("click",()=>sel(x)));
'@
    $enc = { param($t) [System.Net.WebUtility]::HtmlEncode([string]$t) }
    $langDisp = Get-LanguageName $Language
    $langCode = if ($Language) { " ($Language)" } else { '' }
    $cjkCls = if ($Cjk) { ' cjk' } else { '' }
    $tEsc = & $enc $Transcript
    $xEsc = & $enc $Translation
    $trTab = & $enc $TranslationLabel
    if ($Summary) {
        $sumInner = '<div class="body">' + ((& $enc $Summary) -replace "`n", '<br>') + '</div>'
    } else {
        $note = if ($SummaryNote) { $SummaryNote } else { 'Summary unavailable (no local LLM server reachable).' }
        $sumInner = '<div class="note">' + (& $enc $note) + '</div>'
    }
    $wc = '{0:N0}' -f $WordCount
    $dur = & $enc (Format-SpeechDuration $SpeechDurationS)
    $tt = & $enc $Title
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append("<title>$tt</title>`n<style>$css</style>`n")
    [void]$sb.Append('<div class="wrap">' + "`n")
    [void]$sb.Append('<p class="kicker">plaude-local &middot; speech-to-text</p>' + "`n")
    [void]$sb.Append("<h1>$tt</h1>`n")
    [void]$sb.Append('<div class="stats">')
    [void]$sb.Append("<div class=`"stat`"><p class=`"label`">Language detected</p><div class=`"value`">$(& $enc $langDisp)$(& $enc $langCode)</div></div>")
    [void]$sb.Append("<div class=`"stat`"><p class=`"label`">Speech duration</p><div class=`"value`">$dur</div></div>")
    [void]$sb.Append("<div class=`"stat`"><p class=`"label`">Words</p><div class=`"value`">$wc</div></div>")
    [void]$sb.Append("</div>`n")
    $provParts = @()
    if ($TranscriptionEngine) { $provParts += "<span>Transcription <b>$(& $enc $TranscriptionEngine)</b></span>" }
    if ($TranslationEngine) { $provParts += "<span>Translation <b>$(& $enc $TranslationEngine)</b></span>" }
    if ($provParts.Count -gt 0) { [void]$sb.Append("<div class=`"prov`">$($provParts -join '')</div>`n") }
    [void]$sb.Append("<section class=`"summary`"><h2>Critical topics</h2>$sumInner</section>`n")
    [void]$sb.Append('<div class="tabs" role="tablist" aria-label="Views">')
    [void]$sb.Append('<button class="tab" role="tab" id="tab-transcribed" aria-controls="panel-transcribed" aria-selected="true">Transcribed</button>')
    [void]$sb.Append("<button class=`"tab`" role=`"tab`" id=`"tab-translation`" aria-controls=`"panel-translation`" aria-selected=`"false`">$trTab</button>")
    [void]$sb.Append('<button class="tab" role="tab" id="tab-sbs" aria-controls="panel-sbs" aria-selected="false">Side-by-Side</button>')
    [void]$sb.Append("</div>`n")
    [void]$sb.Append("<section class=`"panel`" id=`"panel-transcribed`" role=`"tabpanel`"><div class=`"doc$cjkCls`"><pre>$tEsc</pre></div></section>`n")
    [void]$sb.Append("<section class=`"panel`" id=`"panel-translation`" role=`"tabpanel`" hidden><div class=`"doc`"><pre>$xEsc</pre></div></section>`n")
    # Side-by-Side: time-aligned segment rows (original left, translation right).
    $rows = if ($Pairs) { @($Pairs) } else { @([pscustomobject]@{ O = $Transcript; X = $Translation }) }
    [void]$sb.Append("<section class=`"panel`" id=`"panel-sbs`" role=`"tabpanel`" hidden><div class=`"sbs`">")
    [void]$sb.Append("<div class=`"sbs-h`">Transcribed &middot; $(& $enc $langDisp)</div><div class=`"sbs-h`">$trTab</div>")
    foreach ($row in $rows) {
        $oCell = if (([string]$row.O).Trim()) { & $enc $row.O } else { '&nbsp;' }
        if (([string]$row.X).Trim()) {
            $xCell = "<div class=`"sbs-x`">$(& $enc $row.X)</div>"
        } else {
            $xCell = '<div class="sbs-x empty">&#8212;</div>'
        }
        [void]$sb.Append("<div class=`"sbs-o$cjkCls`">$oCell</div>$xCell")
    }
    [void]$sb.Append("</div></section>`n")
    [void]$sb.Append("<footer>plaude-local &middot; $wc words &middot; $dur</footer>`n")
    [void]$sb.Append("</div>`n<script>$js</script>`n")
    return $sb.ToString()
}

function Invoke-Main {
    if ($Version) { Write-Host "plaude-local (PowerShell) $($Script:ToolVersion)"; return 0 }
    if ($Check) { return (Invoke-Check) }
    if ($ListModels) { return (Invoke-ListModels) }

    Disable-HfTelemetry
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

    # -DiarizeBackend whisperx isn't implemented in the PowerShell port (pyannote
    # only); note it and proceed with pyannote (parity note in powershell/README).
    if ($Diarize -and $DiarizeBackend -eq 'whisperx') {
        Write-Log 'note: -DiarizeBackend whisperx is not implemented in the PowerShell port; using pyannote.'
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

        # Parse whisper-ctranslate2's JSON (per-segment timings + quality metrics).
        $jsonPath = Join-Path $work ([System.IO.Path]::GetFileNameWithoutExtension($prepared) + '.json')
        $report = $null
        $jsonObj = $null
        $segs = @()
        if (Test-Path -LiteralPath $jsonPath) {
            $jsonObj = (Get-Content -LiteralPath $jsonPath -Raw -Encoding UTF8) | ConvertFrom-Json
            $segs = if ($jsonObj.PSObject.Properties['segments']) { $jsonObj.segments } else { @() }
        }

        # Speaker diarization (parity with Python): run pyannote directly, merge
        # turns onto the segments, then rebuild the transcript with speaker labels.
        if ($Diarize) {
            Write-Log '[3/4] diarizing speakers (pyannote) ...'
            try {
                $turns = Invoke-PyannoteDiarize -AudioPath $prepared -Token $token -Device $dev `
                    -Model $DiarizeModel -Num $NumSpeakers -Min $MinSpeakers -Max $MaxSpeakers
                $segs = Merge-Turns -Segments $segs -Turns $turns
                $nspk = (@($segs | ForEach-Object { $_.speaker } | Where-Object { $_ } | Sort-Object -Unique)).Count
                Write-Log "      $nspk speaker(s) detected"
                $transcriptText = Format-DiarizedTranscript -Segments $segs -Fmt $Format
            } catch { Write-ErrLine "error: $($_.Exception.Message)"; return 7 }
        } else {
            Write-Log '[3/4] diarization skipped'
        }

        # Assess recording quality from the JSON metrics whisper-ctranslate2 wrote.
        if (@($segs).Count -gt 0) {
            $report = Get-QualityReport -Segments $segs -Duration (Get-MediaDuration $InputFile) -Thresholds (Get-Thresholds)
            Write-Log ('      ' + (Format-QualitySummary $report))
        }

        # A bad recording triggers the -OnBad policy (default: warn and continue).
        if ($report -and $report.verdict -eq 'bad') {
            if ($OnBad -eq 'skip') { Write-ErrLine ('skipped: bad recording - ' + (Format-QualitySummary $report)); return 0 }
            if ($OnBad -eq 'fail') { Write-ErrLine ('error: bad recording - ' + (Format-QualitySummary $report)); return 12 }
            Write-ErrLine ('warning: ' + (Format-QualitySummary $report) + ' (writing anyway; use -OnBad to change)')
        }

        # 4a. HTML dashboard (default): summary + Transcribed / Translated /
        # Side-by-Side tabs, plus optional split text files.
        if ($Format -eq 'html') {
            $srcLang = if ($jsonObj -and $jsonObj.PSObject.Properties['language']) { [string]$jsonObj.language } else { '' }
            $speechS = $null
            if ($segs -and @($segs).Count -gt 0) {
                $speechS = ($segs | ForEach-Object { [double]$_.end - [double]$_.start } | Measure-Object -Sum).Sum
            }
            if (-not $speechS) { $speechS = Get-MediaDuration $InputFile }

            $target = $TranslateTo.ToLower()
            $translationText = ''
            $pairs = $null   # time-aligned Side-by-Side rows
            $engine = Resolve-TranslateEngine
            $sep = [char]0x00B7
            $transcriptionEngine = "whisper-ctranslate2 $sep $Model"
            $translationEngine = ''

            # Resolve the LLM backend/model/url (summary + LLM translation).
            $sBackend = $SummarizeBackend
            $oUrl = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultOllamaUrl }
            $lUrl = if ($SummarizeUrl) { $SummarizeUrl } else { $Script:DefaultLlamacppUrl }
            if ($sBackend -eq 'auto') { $sBackend = Get-SummBackend -OllamaUrl $oUrl -LlamacppUrl $lUrl }
            $sUrl = if ($sBackend -eq 'llamacpp') { $lUrl } else { $oUrl }
            $sModel = if ($SummarizeModel) { $SummarizeModel } elseif ($sBackend -eq 'ollama') { Get-DefaultOllamaModel -Url $oUrl } else { $null }
            $tModel = if ($TranslateModel) { $TranslateModel } elseif ($sBackend -eq 'ollama') { Get-DefaultOllamaModel -Url $oUrl } else { $null }
            $segTexts = @($segs | ForEach-Object { ([string]$_.text).Trim() })

            if ($target -eq $srcLang.ToLower()) {
                # Target language == source: no translation needed.
                $translationLabel = if ($target) { Get-LanguageName $target } else { 'Original' }
                $translationText = $transcriptText
                $pairs = Get-AlignedPairs -OrigSegments $segs -TransSegments $segs
                $translationEngine = 'none (same as source)'
            } elseif ($engine -eq 'whisper' -and $target -eq 'en') {
                $translationLabel = 'English'
                Write-Log '      translating to English (Whisper) ...'
                $trWork = Join-Path $work 'translate'
                New-Item -ItemType Directory -Force -Path $trWork | Out-Null
                try {
                    $trPath = Invoke-Transcribe -AudioPath $prepared -WorkDir $trWork -Dev $dev -Compute $compute -Token $token -Task translate
                    $translationText = [System.IO.File]::ReadAllText($trPath, [System.Text.Encoding]::UTF8)
                    $trJson = [System.IO.Path]::ChangeExtension($trPath, '.json')
                    if (Test-Path -LiteralPath $trJson) {
                        $trObj = (Get-Content -LiteralPath $trJson -Raw -Encoding UTF8) | ConvertFrom-Json
                        $trSegs = if ($trObj.PSObject.Properties['segments']) { $trObj.segments } else { @() }
                        $pairs = Get-AlignedPairs -OrigSegments $segs -TransSegments $trSegs
                    }
                } catch { Write-ErrLine "error: $($_.Exception.Message)"; return 6 }
                $translationEngine = "Whisper translate $sep $Model"
            } else {
                # LLM (accuracy-first), aligned line-by-line so Side-by-Side lines up.
                $translationLabel = if ($target -eq 'en') { 'English' } else { Get-LanguageName $target }
                if ($sBackend) {
                    Write-Log "      translating to $translationLabel (LLM) ..."
                    try {
                        $translated = Invoke-TranslateLines -Lines $segTexts -TargetLanguage $translationLabel -Backend $sBackend -Model $tModel -Url $sUrl -MaxChars $SummarizeMaxChars -TimeoutSec $SummarizeTimeout
                        $translationText = (($translated | Where-Object { $_ }) -join "`n").Trim()
                        $rowList = New-Object System.Collections.Generic.List[object]
                        for ($i = 0; $i -lt $segTexts.Count; $i++) { $rowList.Add([pscustomobject]@{ O = $segTexts[$i]; X = $translated[$i] }) }
                        $pairs = $rowList.ToArray()
                        $translationEngine = "LLM $sep $(if ($tModel) { $tModel } else { 'default' })"
                    } catch { $translationText = ''; $translationEngine = "LLM $sep unavailable"; Write-Log "      translation unavailable: $($_.Exception.Message)" }
                } else {
                    $translationEngine = "LLM $sep unavailable"
                }
            }

            $summaryText = $null; $summaryNote = $null
            Write-Log '      summarizing critical topics ...'
            if ($sBackend) {
                try { $summaryText = Invoke-Summarize -Text $transcriptText -Backend $sBackend -Model $sModel -Url $sUrl -MaxChars $SummarizeMaxChars -Topics -TimeoutSec $SummarizeTimeout }
                catch { $summaryNote = "Summary unavailable ($($_.Exception.Message))." }
            } else {
                $summaryNote = 'Summary unavailable (no local LLM server reachable).'
            }

            $cjk = $srcLang.ToLower() -in @('zh', 'ja', 'ko')
            $title = [System.IO.Path]::GetFileNameWithoutExtension($InputFile) + ' - transcript dashboard'
            $htmlDoc = Get-DashboardHtml -Title $title -Language $srcLang -SpeechDurationS $speechS `
                -WordCount (Get-WordCount $transcriptText) -Summary $summaryText -SummaryNote $summaryNote `
                -Transcript $transcriptText -Translation $translationText `
                -TranslationLabel "Translated-$translationLabel" -Cjk $cjk -Pairs $pairs `
                -TranscriptionEngine $transcriptionEngine -TranslationEngine $translationEngine

            $dashOut = if ($Output) { $Output } else { Get-DefaultOutput -Fmt 'html' }
            if ($dashOut -eq '-') {
                [Console]::Out.Write($htmlDoc)
            } else {
                if (-not (Confirm-Overwrite -Path $dashOut)) { Write-ErrLine "aborted: $dashOut was not overwritten."; return 11 }
                try { Write-Utf8File -Path $dashOut -Text $htmlDoc }
                catch { Write-ErrLine "error: could not write dashboard to ${dashOut}: $($_.Exception.Message)"; return 9 }
                Write-Log "[4/4] dashboard -> $dashOut"
            }

            if ($SplitOutputs -or $TranscriptionFile -or $TranslationFile) {
                $tFile = if ($TranscriptionFile) { $TranscriptionFile } else { 'transcription.txt' }
                $xFile = if ($TranslationFile) { $TranslationFile } else { 'translation.txt' }
                try {
                    Write-Utf8File -Path $tFile -Text ($transcriptText.TrimEnd("`n") + "`n")
                    Write-Utf8File -Path $xFile -Text (([string]$translationText).TrimEnd("`n") + "`n")
                } catch { Write-ErrLine "error: could not write split outputs: $($_.Exception.Message)"; return 9 }
                Write-Log "      transcription -> $tFile   translation -> $xFile"
            }
            return 0
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
            # Resolve the model like the HTML path / Python _resolve_call: an
            # explicit -SummarizeModel wins, else the first installed Ollama model
            # (llama.cpp binds its own model at launch, so leave it null there).
            $model = if ($SummarizeModel) { $SummarizeModel }
                     elseif ($backend -eq 'ollama') { Get-DefaultOllamaModel -Url $url }
                     else { $null }

            try {
                $summary = Invoke-Summarize -Text $plainText -Backend $backend -Model $model -Url $url -MaxChars $SummarizeMaxChars -TimeoutSec $SummarizeTimeout
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
