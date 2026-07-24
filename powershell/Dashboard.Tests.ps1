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
