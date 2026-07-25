#requires -Version 5.1
<#
    Pester tests for the HTML dashboard renderer + helpers in Plaude-Local.ps1
    (parity with tests/test_dashboard.py). Pure/offline.
#>

BeforeAll {
    . (Join-Path $PSScriptRoot 'Plaude-Local.ps1')
    # CJK sample built from code points so the test file stays ASCII-only.
    $script:CJK = [char]0x4f60 + [char]0x597d + [char]0x4e16 + [char]0x754c   # 4 chars
}

Describe 'Get-WordCount' {
    It 'counts whitespace-delimited words' {
        Get-WordCount 'hello there friend' | Should -Be 3
        Get-WordCount '' | Should -Be 0
    }
    It 'counts each CJK character plus spaced words' {
        Get-WordCount ($script:CJK + ' world') | Should -Be 5
    }
}

Describe 'Format-SpeechDuration' {
    It 'formats seconds / minutes / hours' {
        Format-SpeechDuration 5 | Should -Be '5s'
        Format-SpeechDuration 75 | Should -Be '1m 15s'
        Format-SpeechDuration 3661 | Should -Be '1h 01m 01s'
    }
    It 'returns an em dash for null' {
        Format-SpeechDuration $null | Should -Be ([char]0x2014)
    }
}

Describe 'Get-LanguageName' {
    It 'maps known codes and falls back to the code' {
        Get-LanguageName 'en' | Should -Be 'English'
        Get-LanguageName 'zh' | Should -Be 'Chinese'
        Get-LanguageName 'xx' | Should -Be 'xx'
        Get-LanguageName '' | Should -Be 'Unknown'
    }
}

Describe 'Get-CriticalTopicsPrompt' {
    It 'asks for <=250 words of critical topics and preserves language' {
        $p = Get-CriticalTopicsPrompt -Text 'hi'
        $p | Should -Match '250 words'
        $p | Should -Match 'critical topics'
        $p | Should -Match 'Preserve the original language'
    }
    It 'has a distinct combine variant' {
        Get-CriticalTopicsPrompt -Text 'x' -Combine | Should -Match 'partial notes'
    }
}

Describe 'Build-TranslatePrompt' {
    It 'names the target language and asks for translation only' {
        $p = Build-TranslatePrompt -Text 'hola' -TargetLanguage 'English'
        $p | Should -Match 'Translate the following text into English'
        $p | Should -Match 'ONLY the translation'
    }
}

Describe 'Get-DashboardHtml' {
    BeforeAll {
        $script:Doc = Get-DashboardHtml -Title 'My Clip' -Language 'zh' -SpeechDurationS 754 `
            -WordCount 1234 -Summary 'Critical topic one.' -SummaryNote '' `
            -Transcript $script:CJK -Translation 'Hello world' `
            -TranslationLabel 'Translated-English' -Cjk $true
    }
    It 'has all three tabs and the translated label' {
        $script:Doc | Should -Match 'tab-transcribed'
        $script:Doc | Should -Match 'tab-translation'
        $script:Doc | Should -Match 'tab-sbs'
        $script:Doc | Should -Match 'Side-by-Side'
        $script:Doc | Should -Match 'Translated-English'
    }
    It 'shows stats and the summary' {
        $script:Doc | Should -Match 'Chinese'
        $script:Doc | Should -Match '12m 34s'
        $script:Doc | Should -Match '1,234'
        $script:Doc | Should -Match 'Critical topic one'
    }
    It 'HTML-encodes the transcript (no raw angle brackets injected)' {
        $doc = Get-DashboardHtml -Title 'T' -Language 'en' -SpeechDurationS 5 -WordCount 1 `
            -Summary 's' -SummaryNote '' -Transcript '<script>x</script>' -Translation 'y' `
            -TranslationLabel 'Translated-English' -Cjk $false
        $doc | Should -Not -Match '<script>x</script>'
        $doc | Should -Match '&lt;script&gt;'
    }
    It 'renders a summary note when no summary is available' {
        $doc = Get-DashboardHtml -Title 'T' -Language 'en' -SpeechDurationS 5 -WordCount 1 `
            -Summary '' -SummaryNote 'Summary unavailable (test).' -Transcript 'a' -Translation 'b' `
            -TranslationLabel 'Translated-English' -Cjk $false
        $doc | Should -Match 'Summary unavailable \(test\)\.'
    }
    It 'is self-contained (no external resources)' {
        $script:Doc | Should -Not -Match 'http://'
        $script:Doc | Should -Not -Match 'src='
        $script:Doc | Should -Match '<style>'
    }
    It 'renders aligned Side-by-Side rows from pairs' {
        $pairs = @([pscustomobject]@{ O = 'hola'; X = 'hello' },
                   [pscustomobject]@{ O = 'mundo'; X = 'world' })
        $doc = Get-DashboardHtml -Title 'T' -Language 'es' -SpeechDurationS 5 -WordCount 1 `
            -Summary 's' -SummaryNote '' -Transcript 'x' -Translation 'y' `
            -TranslationLabel 'Translated-English' -Cjk $false -Pairs $pairs
        $doc | Should -Match 'class="sbs-o'
        $doc | Should -Match 'class="sbs-x'
        ([regex]::Matches($doc, 'class="sbs-o')).Count | Should -Be 2
        $doc | Should -Match 'hello'
        $doc | Should -Match 'world'
    }
}

Describe 'Get-AlignedPairs' {
    It 'pairs original + translation segments by time overlap' {
        $orig = @([pscustomobject]@{ start = 0.0; end = 2.0; text = 'a' },
                  [pscustomobject]@{ start = 2.0; end = 4.0; text = 'b' })
        $trans = @([pscustomobject]@{ start = 0.1; end = 1.9; text = 'A' },
                   [pscustomobject]@{ start = 2.1; end = 3.9; text = 'B' })
        $rows = @(Get-AlignedPairs -OrigSegments $orig -TransSegments $trans)
        $rows.Count | Should -Be 2
        $rows[0].O | Should -Be 'a'
        $rows[0].X | Should -Be 'A'
        $rows[1].X | Should -Be 'B'
    }
    It 'leaves an empty translation cell when unmatched' {
        $orig = @([pscustomobject]@{ start = 0.0; end = 1.0; text = 'a' },
                  [pscustomobject]@{ start = 5.0; end = 6.0; text = 'b' })
        $trans = @([pscustomobject]@{ start = 0.2; end = 0.8; text = 'A' })
        $rows = @(Get-AlignedPairs -OrigSegments $orig -TransSegments $trans)
        $rows[1].X | Should -Be ''
    }
}

Describe 'Invoke-TranslateLines' {
    It 'parses numbered lines, strips reasoning, keeps alignment' {
        Mock Invoke-LlmCall { "<think>plan</think>`n1. Hello`n2. World" }
        # Direct assignment (as the CLI uses it) unwraps the returned string[].
        $out = Invoke-TranslateLines -Lines @('a', 'b') -TargetLanguage 'English' -Backend 'ollama' -Model 'm' -Url 'http://x' -MaxChars 8000
        $out.Count | Should -Be 2
        $out[0] | Should -Be 'Hello'
        $out[1] | Should -Be 'World'
    }
    It 'leaves unmatched lines empty' {
        Mock Invoke-LlmCall { '1. Hi' }
        $out = Invoke-TranslateLines -Lines @('a', 'b') -TargetLanguage 'English' -Backend 'ollama' -Model 'm' -Url 'http://x' -MaxChars 8000
        $out[1] | Should -Be ''
    }
    It 'forwards the request timeout to the LLM call' {
        Mock Invoke-LlmCall { '1. Hi' }
        Invoke-TranslateLines -Lines @('a') -TargetLanguage 'English' -Backend 'ollama' -Model 'm' -Url 'http://x' -MaxChars 8000 -TimeoutSec 600 | Out-Null
        Should -Invoke Invoke-LlmCall -Times 1 -ParameterFilter { $TimeoutSec -eq 600 }
    }
    It 'caps lines per request (large batches would overflow the context)' {
        Mock Invoke-LlmCall {
            $n = ([regex]::Matches($Prompt, '(?m)^\d+\. ')).Count
            (1..$n | ForEach-Object { "$_. t$_" }) -join "`n"
        }
        $lines = 0..99 | ForEach-Object { "L$_" }
        $out = Invoke-TranslateLines -Lines $lines -TargetLanguage 'English' -Backend 'ollama' -Model 'm' -Url 'http://x' -MaxChars 100000 -MaxLines 40
        $out.Count | Should -Be 100
        ($out | Where-Object { $_ }).Count | Should -Be 100      # nothing dropped
        Should -Invoke Invoke-LlmCall -Times 3                    # 100 / 40 -> 3 batches
    }
    It 'splits and retries when a batch under-parses (never all-empty)' {
        Mock Invoke-LlmCall {
            $n = ([regex]::Matches($Prompt, '(?m)^\d+\. ')).Count
            if ($n -gt 3) { 'sorry, cannot' } else { (1..$n | ForEach-Object { "$_. ok$_" }) -join "`n" }
        }
        $lines = 0..7 | ForEach-Object { "L$_" }
        $out = Invoke-TranslateLines -Lines $lines -TargetLanguage 'English' -Backend 'ollama' -Model 'm' -Url 'http://x' -MaxChars 100000 -MaxLines 8
        $out.Count | Should -Be 8
        ($out | Where-Object { $_ }).Count | Should -Be 8
    }
}

Describe 'Invoke-LlmCall num_ctx' {
    It 'sends a fixed ollama context window above the truncating default' {
        Mock Invoke-RestMethod { [pscustomobject]@{ response = 'ok' } }
        # Same fixed value regardless of prompt length (a varying num_ctx would
        # force Ollama to reload the model every call).
        Invoke-LlmCall -Prompt ('x' * 20000) -Backend 'ollama' -Model 'm' -Url 'http://x' | Out-Null
        Invoke-LlmCall -Prompt 'short' -Backend 'ollama' -Model 'm' -Url 'http://x' | Out-Null
        Should -Invoke Invoke-RestMethod -Times 2 -ParameterFilter {
            $obj = [System.Text.Encoding]::UTF8.GetString($Body) | ConvertFrom-Json
            $obj.options.num_ctx -eq 8192
        }
    }
}

Describe 'Resolve-TranslateEngine' {
    It 'returns the explicit choice' {
        $TranslateEngine = 'whisper'
        Resolve-TranslateEngine | Should -Be 'whisper'
    }
    It 'auto resolves to llm when a backend is reachable' {
        $TranslateEngine = 'auto'
        Mock Get-SummBackend { 'ollama' }
        Resolve-TranslateEngine | Should -Be 'llm'
    }
    It 'auto resolves to whisper when no backend' {
        $TranslateEngine = 'auto'
        Mock Get-SummBackend { $null }
        Resolve-TranslateEngine | Should -Be 'whisper'
    }
}

Describe 'Get-DefaultOllamaModel' {
    It 'returns the first installed model' {
        Mock Invoke-RestMethod { [pscustomobject]@{ models = @([pscustomobject]@{ name = 'gemma4:latest' }, [pscustomobject]@{ name = 'llama3.1' }) } }
        Get-DefaultOllamaModel | Should -Be 'gemma4:latest'
    }
    It 'returns null when the server is unreachable' {
        Mock Invoke-RestMethod { throw 'down' }
        Get-DefaultOllamaModel | Should -BeNullOrEmpty
    }
}

Describe 'Default local LLM URLs' {
    It 'use 127.0.0.1, not localhost (avoids the IPv6 ::1 detection stall)' {
        # Regression: Invoke-RestMethod stalls on localhost->::1 and times out the
        # 2s backend detection, silently disabling translation/summary. Parity
        # with plaude_local.summarize DEFAULT_*_URL.
        $Script:DefaultOllamaUrl | Should -Be 'http://127.0.0.1:11434'
        $Script:DefaultLlamacppUrl | Should -Be 'http://127.0.0.1:8080'
        $Script:DefaultOllamaUrl | Should -Not -Match 'localhost'
    }
}

Describe 'Get-OllamaModels' {
    It 'lists all installed model names' {
        Mock Invoke-RestMethod { [pscustomobject]@{ models = @([pscustomobject]@{ name = 'qwen2.5:7b' }, [pscustomobject]@{ name = 'gemma4:latest' }) } }
        $m = @(Get-OllamaModels)
        $m.Count | Should -Be 2
        $m[0] | Should -Be 'qwen2.5:7b'
    }
    It 'returns an empty list when the server is unreachable' {
        Mock Invoke-RestMethod { throw 'down' }
        @(Get-OllamaModels).Count | Should -Be 0
    }
}

Describe 'Invoke-ListModels' {
    It 'lists models and marks the first as the translation default' {
        Mock Get-OllamaModels { @('qwen2.5:7b', 'gemma4:latest') }
        $out = (Invoke-ListModels 6>&1) -join "`n"
        $out | Should -Match 'qwen2.5:7b'
        $out | Should -Match 'gemma4:latest'
        ($out -split "`n" | Where-Object { $_ -match 'default for translation' }) | Should -Match 'qwen2.5:7b'
    }
    It 'marks an explicit -TranslateModel as the default' {
        Mock Get-OllamaModels { @('qwen2.5:7b', 'gemma4:latest') }
        $TranslateModel = 'gemma4:latest'
        $out = (Invoke-ListModels 6>&1) -join "`n"
        ($out -split "`n" | Where-Object { $_ -match 'default for translation' }) | Should -Match 'gemma4:latest'
    }
    It 'returns 1 with a remedy when no models are installed' {
        Mock Get-OllamaModels { @() }
        $rc = Invoke-ListModels 6>$null
        $rc | Should -Be 1
    }
}

Describe 'Get-DashboardHtml provenance' {
    It 'renders the transcription and translation engines' {
        $doc = Get-DashboardHtml -Title 'T' -Language 'en' -SpeechDurationS 5 -WordCount 1 `
            -Summary 's' -SummaryNote '' -Transcript 'a' -Translation 'b' `
            -TranslationLabel 'Translated-English' -Cjk $false -Pairs $null `
            -TranscriptionEngine 'whisper-ctranslate2 ENG' -TranslationEngine 'LLM MODELX'
        $doc | Should -Match 'class="prov"'
        $doc | Should -Match 'whisper-ctranslate2 ENG'
        $doc | Should -Match 'LLM MODELX'
    }
}
