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
}

Describe 'Build-SummaryPrompt' {
    It 'preserves the original language in the instruction' {
        Build-SummaryPrompt -Text 'hi' | Should -Match 'Preserve the original language'
    }
    It 'has a distinct combine variant' {
        Build-SummaryPrompt -Text 'x' -Combine | Should -Match 'partial summaries'
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
