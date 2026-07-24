#requires -Version 5.1
<#
    Pester tests for Plaude-Local.ps1 (PowerShell edition).

    These mirror the offline parts of the Python unittest suite: pure/logic
    functions only, no external tools (ffmpeg, whisper-ctranslate2) and no
    network. The script is dot-sourced so its functions can be tested in
    isolation - its main() does not run when dot-sourced.

    Run:  Invoke-Pester -Path ./powershell
#>

BeforeAll {
    . (Join-Path $PSScriptRoot 'Plaude-Local.ps1')
    # Defaults some functions read from the script's param scope.
    $ComputeType = 'auto'
    $Yes = $false
    $Quiet = $true
    $NoVad = $false
    $Summarize = $false
    $Format = 'txt'
}

Describe 'Get-DefaultOutput' {
    It 'defaults to output.<format>' {
        Get-DefaultOutput -Fmt 'txt' | Should -Be 'output.txt'
        Get-DefaultOutput -Fmt 'srt' | Should -Be 'output.srt'
        Get-DefaultOutput -Fmt 'json' | Should -Be 'output.json'
    }
}

Describe 'Resolve-ComputeType' {
    BeforeEach { $ComputeType = 'auto' }

    It 'uses float16 on CUDA' {
        Resolve-ComputeType -Dev 'cuda' | Should -Be 'float16'
    }
    It 'uses int8 on CPU' {
        Resolve-ComputeType -Dev 'cpu' | Should -Be 'int8'
    }
    It 'passes an explicit compute type through' {
        $ComputeType = 'int8_float16'
        Resolve-ComputeType -Dev 'cuda' | Should -Be 'int8_float16'
    }
}

Describe 'Resolve-Device' {
    It 'passes an explicit device through without probing (parity with Python)' {
        $Device = 'cpu'
        Resolve-Device | Should -Be 'cpu'
        $Device = 'cuda'
        Resolve-Device | Should -Be 'cuda'
    }
    It 'cpu-only mode stays cpu and never probes for a GPU' {
        # nvidia-smi must not even be looked up when the device is explicit.
        Mock Test-Command { throw 'explicit device must not probe for a GPU' }
        $Device = 'cpu'
        Resolve-Device | Should -Be 'cpu'
        Should -Invoke Test-Command -Times 0 -Exactly
    }
    It 'gpu-only mode stays cuda without probing or fallback' {
        Mock Test-Command { throw 'explicit device must not probe for a GPU' }
        $Device = 'cuda'
        Resolve-Device | Should -Be 'cuda'
        Should -Invoke Test-Command -Times 0 -Exactly
    }
}

Describe 'Split-IntoChunks' {
    It 'returns a single chunk for short text' {
        (Split-IntoChunks -Text 'hello world' -MaxChars 100).Count | Should -Be 1
    }
    It 'splits long text into multiple chunks' {
        $text = (1..100 | ForEach-Object { "line $_" }) -join "`n"
        (Split-IntoChunks -Text $text -MaxChars 50).Count | Should -BeGreaterThan 1
    }
    It 'does not hang and loses no characters for pathological MaxChars' {
        # MaxChars <= 1 must terminate (regression parity with Python).
        $joined = (Split-IntoChunks -Text 'abcdef' -MaxChars 0) -join ''
        $joined | Should -Be 'abcdef'
    }
    It 'produces no empty chunks' {
        $text = "aaaa`n`n`n`n`n`nbbbb`n`n`n`n`n`ncccc"
        foreach ($c in (Split-IntoChunks -Text $text -MaxChars 5)) {
            $c.Trim() | Should -Not -BeNullOrEmpty
        }
    }
    It 'uses floor(MaxChars/2) as the break threshold (parity with Python //)' {
        # With MaxChars=7 the break threshold is floor(7/2)=3. A newline at
        # index 3 sits exactly AT the threshold, so it IS accepted as a break
        # point and the first chunk is "abc". The old [int](7/2)=4 (banker's
        # rounding) would reject it and hard-split, yielding "abc`ndef".
        $chunks = Split-IntoChunks -Text "abc`ndefghij" -MaxChars 7
        $chunks[0] | Should -Be 'abc'
    }
}

Describe 'Build-SummaryPrompt' {
    It 'preserves the original language in the instruction' {
        Build-SummaryPrompt -Text 'hi' | Should -Match 'Preserve the original language'
    }
    It 'has a distinct combine variant' {
        Build-SummaryPrompt -Text 'x' -Combine | Should -Match 'partial summaries'
    }
}

Describe 'Invoke-Summarize (map-reduce)' {
    It 'chunks long text, summarizes each chunk, then combines (parity with Python)' {
        # Mock the transport; record how many calls happen and whether a combine
        # prompt was used. A hashtable (reference type) survives the mock scope.
        $rec = @{ count = 0; combined = $false }
        Mock Invoke-LlmCall {
            $rec.count++
            if ($Prompt -match 'partial summaries') { $rec.combined = $true }
            return 'a partial summary sentence'
        }
        $long = (1..200 | ForEach-Object { "sentence number $_ here" }) -join "`n"
        $out = Invoke-Summarize -Text $long -Backend 'ollama' -Model 'm' -Url 'http://x' -MaxChars 100
        $out | Should -Not -BeNullOrEmpty
        $rec.count | Should -BeGreaterThan 1          # fanned out over chunks
        $rec.combined | Should -BeTrue                # a combine pass ran
    }

    It 'summarizes short text in a single call' {
        $rec = @{ count = 0 }
        Mock Invoke-LlmCall { $rec.count++; return 'summary' }
        $out = Invoke-Summarize -Text 'a short transcript' -Backend 'ollama' -Model 'm' -Url 'http://x' -MaxChars 1000
        $out | Should -Be 'summary'
        $rec.count | Should -Be 1
    }

    It 'throws on empty transcript (parity with Python)' {
        { Invoke-Summarize -Text '   ' -Backend 'ollama' -Model 'm' -Url 'http://x' -MaxChars 100 } |
            Should -Throw
    }
}

Describe 'Test-Command' {
    It 'finds an existing command' {
        # pwsh is running these tests, so it must resolve.
        Test-Command 'pwsh' | Should -BeTrue
    }
    It 'reports a missing command as false' {
        Test-Command 'definitely-not-a-real-command-xyz' | Should -BeFalse
    }
}

Describe 'Confirm-Overwrite' {
    It 'returns true for a new (non-existent) file' {
        $Yes = $false
        Confirm-Overwrite -Path (Join-Path $TestDrive 'does-not-exist.txt') |
            Should -BeTrue
    }
    It 'returns true immediately when -Yes is set, even if the file exists' {
        $existing = Join-Path $TestDrive 'exists.txt'
        Set-Content -LiteralPath $existing -Value 'old'
        $Yes = $true
        Confirm-Overwrite -Path $existing | Should -BeTrue
    }
}

Describe 'Write-Utf8File' {
    It 'round-trips non-Latin (CJK) text as UTF-8 without BOM' {
        $p = Join-Path $TestDrive 'cjk.txt'
        Write-Utf8File -Path $p -Text "你好世界 こんにちは"
        $bytes = [System.IO.File]::ReadAllBytes($p)
        # No UTF-8 BOM (EF BB BF).
        ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) |
            Should -BeFalse
        [System.Text.Encoding]::UTF8.GetString($bytes) | Should -Match '你好世界'
    }
}

Describe 'Get-EnhanceFilters' {
    It 'returns null for none with no gain' {
        Get-EnhanceFilters -Mode 'none' -GainDb 0.0 | Should -BeNullOrEmpty
    }
    It 'returns just a volume filter for none + gain' {
        Get-EnhanceFilters -Mode 'none' -GainDb 6 | Should -Be 'volume=6dB'
    }
    It 'builds the speech chain' {
        Get-EnhanceFilters -Mode 'speech' -GainDb 0.0 | Should -Match 'speechnorm'
    }
    It 'builds the strong chain with compressor and EQ' {
        $f = Get-EnhanceFilters -Mode 'strong' -GainDb 0.0
        $f | Should -Match 'acompressor'
        $f | Should -Match 'equalizer'
    }
    It 'appends gain to a chain' {
        (Get-EnhanceFilters -Mode 'speech' -GainDb -3) | Should -Match 'volume=-3dB$'
    }
    It 'formats fractional gain with a dot on any locale (invariant)' {
        # Force a comma-decimal culture; the filter must still use '.'.
        $old = [System.Threading.Thread]::CurrentThread.CurrentCulture
        try {
            [System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::GetCultureInfo('de-DE')
            Get-EnhanceFilters -Mode 'none' -GainDb -3.5 | Should -Be 'volume=-3.5dB'
        } finally {
            [System.Threading.Thread]::CurrentThread.CurrentCulture = $old
        }
    }
}

Describe 'Invoke-Prepare' {
    It 'is a single pass when no enhance and no gain' {
        $Denoise = 'none'; $Enhance = 'none'; $Gain = 0.0
        Mock ConvertTo-Wav {}
        $out = Invoke-Prepare -Src (Join-Path $TestDrive 'in.wav') -WorkDir $TestDrive
        $out | Should -Match 'prepared\.wav$'
        Should -Invoke ConvertTo-Wav -Times 1 -Exactly
    }
    It 'uses two passes (denoise -> enhance) for speech' {
        $Denoise = 'none'; $Enhance = 'speech'; $Gain = 0.0
        Mock ConvertTo-Wav {}
        Invoke-Prepare -Src (Join-Path $TestDrive 'in.wav') -WorkDir $TestDrive | Out-Null
        Should -Invoke ConvertTo-Wav -Times 2 -Exactly
    }
    It 'routes deepfilter denoise into an intermediate, then enhances' {
        $Denoise = 'deepfilter'; $Enhance = 'speech'; $Gain = 0.0
        Mock ConvertTo-Wav {}
        Mock Invoke-DeepFilter {}
        Invoke-Prepare -Src (Join-Path $TestDrive 'in.wav') -WorkDir $TestDrive | Out-Null
        Should -Invoke Invoke-DeepFilter -Times 1 -Exactly
        Should -Invoke ConvertTo-Wav -Times 1 -Exactly   # only the enhance pass
    }
    It 'runs resemble then a volume pass when gain is set' {
        $Denoise = 'none'; $Enhance = 'resemble'; $Gain = 6
        Mock ConvertTo-Wav {}
        Mock Invoke-ResembleEnhance {}
        Mock Move-Item {}
        Invoke-Prepare -Src (Join-Path $TestDrive 'in.wav') -WorkDir $TestDrive | Out-Null
        Should -Invoke Invoke-ResembleEnhance -Times 1 -Exactly
        # denoise pass + gain volume pass
        Should -Invoke ConvertTo-Wav -Times 2 -Exactly
    }
}

Describe 'ConvertFrom-FfmpegLevels' {
    BeforeAll {
        # Assign inside BeforeAll (not the Describe body) so the value survives
        # into the It blocks under Pester 5+, where the Describe body runs only
        # at discovery time. Otherwise $stderr is unset at run time and throws
        # under Set-StrictMode -Version Latest.
        $stderr = @'
[Parsed_volumedetect_0 @ 0x1] mean_volume: -23.4 dB
[Parsed_volumedetect_0 @ 0x1] max_volume: -2.1 dB
[silencedetect @ 0x2] silence_end: 3 | silence_duration: 3.0
[silencedetect @ 0x2] silence_duration: 2.0
'@
    }
    It 'parses levels and computes silence ratio' {
        $s = ConvertFrom-FfmpegLevels -Text $stderr -Duration 10.0
        $s.mean_volume_db | Should -Be -23.4
        $s.max_volume_db | Should -Be -2.1
        $s.silence_ratio | Should -Be 0.5
    }
    It 'leaves silence ratio null without a duration' {
        (ConvertFrom-FfmpegLevels -Text $stderr -Duration $null).silence_ratio |
            Should -BeNullOrEmpty
    }
}

Describe 'Get-QualityReport' {
    BeforeEach {
        $MinSpeech = 0.15; $MaxCompression = 2.4; $MinLogprob = -1.0; $MaxNoSpeech = 0.6
    }
    It 'flags a good recording as ok' {
        $segs = @([pscustomobject]@{ start = 0; end = 9; text = 'clear speech';
            no_speech_prob = 0.03; avg_logprob = -0.3; compression_ratio = 1.4 })
        (Get-QualityReport -Segments $segs -Duration 10.0).verdict | Should -Be 'ok'
    }
    It 'flags an empty transcript as bad' {
        $segs = @([pscustomobject]@{ start = 0; end = 0; text = '';
            no_speech_prob = 0.9; avg_logprob = -2.0; compression_ratio = 3.0 })
        (Get-QualityReport -Segments $segs -Duration 10.0).verdict | Should -Be 'bad'
    }
    It 'flags low confidence as suspect' {
        $segs = @([pscustomobject]@{ start = 0; end = 9; text = 'garbled';
            no_speech_prob = 0.2; avg_logprob = -1.6; compression_ratio = 1.4 })
        (Get-QualityReport -Segments $segs -Duration 10.0).verdict | Should -Be 'suspect'
    }
    It 'does not throw and stays ok when segments lack metric fields' {
        # Parity with Python's dict.get: absent metrics must be treated as null
        # under Set-StrictMode -Version Latest, not throw.
        $segs = @([pscustomobject]@{ start = 0; end = 5; text = 'hello' })
        (Get-QualityReport -Segments $segs -Duration 10.0).verdict | Should -Be 'ok'
    }
    It 'rounds metrics to 4 decimals (parity with Python)' {
        $segs = @([pscustomobject]@{ start = 0; end = 3.33333; text = 'hi';
            no_speech_prob = 0.123456; avg_logprob = -0.5; compression_ratio = 1.4 })
        $r = Get-QualityReport -Segments $segs -Duration 10.0
        $r.metrics.avg_no_speech_prob | Should -Be 0.1235
    }
    It 'flags near-silent audio as bad' {
        $r = Get-QualityReport -AudioStats @{ mean_volume_db = -62.0; silence_ratio = 0.4 }
        $r.verdict | Should -Be 'bad'
    }
    It 'flags mostly-silence audio as suspect' {
        $r = Get-QualityReport -AudioStats @{ mean_volume_db = -18.0; silence_ratio = 0.9 }
        $r.verdict | Should -Be 'suspect'
    }
}

Describe 'Disable-HfTelemetry' {
    BeforeEach {
        $script:sHub = $env:HF_HUB_DISABLE_TELEMETRY
        $script:sGen = $env:DISABLE_TELEMETRY
        Remove-Item Env:HF_HUB_DISABLE_TELEMETRY -ErrorAction SilentlyContinue
        Remove-Item Env:DISABLE_TELEMETRY -ErrorAction SilentlyContinue
    }
    AfterEach {
        if ($null -eq $script:sHub) { Remove-Item Env:HF_HUB_DISABLE_TELEMETRY -ErrorAction SilentlyContinue } else { $env:HF_HUB_DISABLE_TELEMETRY = $script:sHub }
        if ($null -eq $script:sGen) { Remove-Item Env:DISABLE_TELEMETRY -ErrorAction SilentlyContinue } else { $env:DISABLE_TELEMETRY = $script:sGen }
    }
    It 'disables HF telemetry by default' {
        Disable-HfTelemetry
        $env:HF_HUB_DISABLE_TELEMETRY | Should -Be '1'
        $env:DISABLE_TELEMETRY | Should -Be '1'
    }
    It 'respects an explicit opt-in' {
        $env:HF_HUB_DISABLE_TELEMETRY = '0'
        Disable-HfTelemetry
        $env:HF_HUB_DISABLE_TELEMETRY | Should -Be '0'
    }
}

Describe 'Set-HfOffline' {
    BeforeEach {
        $script:savedHub = $env:HF_HUB_OFFLINE
        $script:savedTfm = $env:TRANSFORMERS_OFFLINE
        Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
        Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
    }
    AfterEach {
        if ($null -eq $script:savedHub) { Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue } else { $env:HF_HUB_OFFLINE = $script:savedHub }
        if ($null -eq $script:savedTfm) { Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue } else { $env:TRANSFORMERS_OFFLINE = $script:savedTfm }
    }
    It 'sets nothing when disabled (parity with Python apply_offline)' {
        Set-HfOffline -Enabled $false
        $env:HF_HUB_OFFLINE | Should -BeNullOrEmpty
        $env:TRANSFORMERS_OFFLINE | Should -BeNullOrEmpty
    }
    It 'sets both offline flags when enabled' {
        Set-HfOffline -Enabled $true
        $env:HF_HUB_OFFLINE | Should -Be '1'
        $env:TRANSFORMERS_OFFLINE | Should -Be '1'
    }
}

Describe 'Get-CudaBootstrap' {
    # Parity with the Python entry point: the PowerShell GPU path runs
    # whisper-ctranslate2 through a bootstrap that registers the nvidia-*-cu12
    # wheel DLL dirs before CUDA loads. Guard the constant against edits.
    It 'registers the wheel DLL dirs and runs the whisper-ctranslate2 entry point' {
        $code = Get-CudaBootstrap
        $code | Should -Match 'add_dll_directory'
        $code | Should -Match 'nvidia'
        $code | Should -Match 'from whisper_ctranslate2\.whisper_ctranslate2 import main'
    }
}

Describe 'Resolve-PythonExe' {
    It 'returns python, py, or $null (never throws)' {
        $r = Resolve-PythonExe
        ($null -eq $r -or $r -in @('python', 'py')) | Should -BeTrue
    }
}

Describe 'Script invocation (CLI param binding)' {
    # Regression: a [switch]$Version parameter once collided with a
    # $Script:Version = '0.1.0' constant. At script scope they are the same
    # variable, so assigning the version string to the switch-typed variable
    # made EVERY direct invocation throw "Cannot convert ... to
    # SwitchParameter" before the body ran. The dot-sourced tests above could
    # not catch it, so exercise the real CLI path in a child process here.
    It 'prints the version with -Version without a binding error' {
        $script = Join-Path $PSScriptRoot 'Plaude-Local.ps1'
        $exe = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
        $out = & $exe -NoProfile -ExecutionPolicy Bypass -File $script -Version 2>&1
        $LASTEXITCODE | Should -Be 0
        ($out -join "`n") | Should -Match 'plaude-local \(PowerShell\) \d+\.\d+\.\d+'
    }
}

Describe 'Register-FfmpegPath' {
    It 'accepts a directory containing both binaries and updates PATH' {
        Mock Test-Command { $true }   # post-PATH recheck resolves
        $dir = Join-Path $TestDrive 'ff'
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $dir 'ffmpeg.exe') -Value 'x'
        Set-Content -LiteralPath (Join-Path $dir 'ffprobe.exe') -Value 'x'
        Register-FfmpegPath -Location $dir | Should -BeTrue
        $env:PATH | Should -Match ([regex]::Escape($dir))
    }
    It 'accepts a binary path and uses its parent directory' {
        Mock Test-Command { $true }
        $dir = Join-Path $TestDrive 'ff3'
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        $exe = Join-Path $dir 'ffmpeg.exe'
        Set-Content -LiteralPath $exe -Value 'x'
        Set-Content -LiteralPath (Join-Path $dir 'ffprobe.exe') -Value 'x'
        Register-FfmpegPath -Location $exe | Should -BeTrue
    }
    It 'rejects a directory missing ffprobe' {
        $dir = Join-Path $TestDrive 'ff2'
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $dir 'ffmpeg.exe') -Value 'x'
        Register-FfmpegPath -Location $dir | Should -BeFalse
    }
    It 'rejects a nonexistent directory' {
        Register-FfmpegPath -Location (Join-Path $TestDrive 'nope') | Should -BeFalse
    }
}

Describe 'Confirm-Ffmpeg' {
    BeforeEach {
        $FfmpegLocation = $null; $InstallMissing = $false; $NoProvision = $false; $Quiet = $true
    }
    It 'proceeds when ffmpeg is present' {
        Mock Test-Command { $true }
        Confirm-Ffmpeg | Should -BeTrue
    }
    It 'uses an explicit -FfmpegLocation when valid' {
        Mock Test-Command { $false }
        Mock Register-FfmpegPath { $true }
        $FfmpegLocation = '/opt/ff'
        Confirm-Ffmpeg | Should -BeTrue
    }
    It 'auto-installs when -InstallMissing and the install succeeds' {
        Mock Test-Command { $false }
        Mock Install-Ffmpeg { $true }
        $InstallMissing = $true
        Confirm-Ffmpeg | Should -BeTrue
    }
    It 'aborts when -InstallMissing but the install fails' {
        Mock Test-Command { $false }
        Mock Install-Ffmpeg { $false }
        $InstallMissing = $true
        Confirm-Ffmpeg | Should -BeFalse
    }
    It 'aborts non-interactively without authorization' {
        Mock Test-Command { $false }
        # Pester runs with redirected input -> non-interactive -> abort.
        Confirm-Ffmpeg | Should -BeFalse
    }
    It 'aborts when -NoProvision is set' {
        Mock Test-Command { $false }
        $NoProvision = $true
        Confirm-Ffmpeg | Should -BeFalse
    }
}
