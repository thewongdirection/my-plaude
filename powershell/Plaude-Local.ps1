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

    # audio preprocessing
    [ValidateSet('ffmpeg', 'deepfilter', 'none')]
    [string]$Denoise = 'ffmpeg',
    [string]$KeepClean,

    # diarization
    [switch]$Diarize,
    [ValidateSet('pyannote', 'whisperx')]
    [string]$DiarizeBackend = 'pyannote',
    [string]$HfToken,
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
$Script:DefaultOllamaUrl = 'http://localhost:11434'
$Script:DefaultLlamacppUrl = 'http://localhost:8080'
$Script:DefaultOllamaModel = 'llama3.1'
$Script:OverwriteTimeoutSeconds = 10
$Script:Version = '0.1.0'

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

function Invoke-Prepare {
    param([string]$Src, [string]$WorkDir)
    $dst = Join-Path $WorkDir ([System.IO.Path]::GetFileNameWithoutExtension($Src) + '.prepared.wav')
    switch ($Denoise) {
        'none'       { ConvertTo-Wav -Src $Src -Dst $dst -Filters $null }
        'ffmpeg'     { ConvertTo-Wav -Src $Src -Dst $dst -Filters $Script:FfmpegDenoiseChain }
        'deepfilter' { Invoke-DeepFilter -Src $Src -Dst $dst }
    }
    return $dst
}

# --------------------------------------------------------------------------- #
# Transcription via whisper-ctranslate2 (same faster-whisper/CTranslate2 engine)
# --------------------------------------------------------------------------- #
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

function Invoke-Transcribe {
    param([string]$AudioPath, [string]$WorkDir, [string]$Dev, [string]$Compute, [string]$Token)

    # When summarizing a non-txt format we ask for 'all' so a plain-text copy
    # (for the LLM) is produced alongside the requested format.
    $outFormat = if ($Summarize -and $Format -ne 'txt') { 'all' } else { $Format }

    $a = @(
        $AudioPath,
        '--model', $Model,
        '--device', $Dev,
        '--compute_type', $Compute,
        '--output_dir', $WorkDir,
        '--output_format', $outFormat,
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
    & whisper-ctranslate2 @a 2>&1 | ForEach-Object { Write-Log ([string]$_) }
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
    if ($Version) { Write-Host "plaude-local (PowerShell) $($Script:Version)"; return 0 }
    if ($Check) { return (Invoke-Check) }

    if (-not $InputFile) {
        Write-ErrLine 'error: an input audio file is required (or use -Check to verify your setup).'
        return 2
    }
    if (-not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
        Write-ErrLine "error: input file not found: $InputFile"
        return 2
    }
    if (-not (Test-Command 'ffmpeg')) {
        Write-ErrLine 'error: ffmpeg not found on PATH. Install it: winget install ffmpeg'
        return 3
    }

    # FFmpeg is the arbiter of decodability (parity with the Python version).
    if (Test-Command 'ffprobe') {
        $codec = Get-AudioCodec -Path $InputFile
        if (-not $codec) {
            Write-ErrLine "error: FFmpeg found no decodable audio stream in $InputFile."
            return 10
        }
        Write-Log "      input audio codec: $codec"
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
        Write-Log "[1/4] preparing audio (denoise=$Denoise) ..."
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
