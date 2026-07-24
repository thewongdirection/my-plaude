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
