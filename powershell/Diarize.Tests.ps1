#requires -Version 5.1
<#
    Pester tests for the PowerShell diarization merge logic in Plaude-Local.ps1
    (parity with tests/test_diarize.py). Pure/offline - no pyannote needed.
#>

BeforeAll {
    . (Join-Path $PSScriptRoot 'Plaude-Local.ps1')
    # Turns as Invoke-PyannoteDiarize yields them: objects {Start, End, Label}.
    $script:TURNS = @(
        [pscustomobject]@{ Start = 0.0; End = 3.0; Label = 'SPEAKER_01' },
        [pscustomobject]@{ Start = 3.0; End = 7.0; Label = 'SPEAKER_00' })
}

Describe 'Get-BestSpeaker' {
    It 'picks the max-overlap speaker' {
        Get-BestSpeaker -Start 0.1 -End 2.5 -Turns $script:TURNS | Should -Be 'SPEAKER_01'
        Get-BestSpeaker -Start 3.5 -End 6.0 -Turns $script:TURNS | Should -Be 'SPEAKER_00'
    }
    It 'straddling the boundary picks the larger side' {
        # [2.0,5.0]: 1.0s in SPEAKER_01, 2.0s in SPEAKER_00 -> SPEAKER_00
        Get-BestSpeaker -Start 2.0 -End 5.0 -Turns $script:TURNS | Should -Be 'SPEAKER_00'
    }
    It 'returns null when there is no overlap' {
        Get-BestSpeaker -Start 20.0 -End 25.0 -Turns $script:TURNS | Should -BeNullOrEmpty
    }
}

Describe 'Get-SpeakerMap' {
    It 'numbers labels by earliest start, even when given out of order' {
        $t = @([pscustomobject]@{ Start = 5; End = 6; Label = 'B' },
               [pscustomobject]@{ Start = 0; End = 1; Label = 'A' },
               [pscustomobject]@{ Start = 2; End = 3; Label = 'B' })
        $m = Get-SpeakerMap -Turns $t
        $m['A'] | Should -Be 'Speaker 1'
        $m['B'] | Should -Be 'Speaker 2'
    }
    It 'handles a single speaker' {
        $m = Get-SpeakerMap -Turns @([pscustomobject]@{ Start = 0; End = 1; Label = 'X' })
        $m['X'] | Should -Be 'Speaker 1'
    }
}

Describe 'Merge-Turns' {
    It 'returns segments unchanged when there are no turns' {
        $segs = @([pscustomobject]@{ start = 0.0; end = 1.0; text = 'hi' })
        (Merge-Turns -Segments $segs -Turns @()).Count | Should -Be 1
    }
    It 'tags each segment with its best-overlap speaker (relabelled)' {
        $segs = @([pscustomobject]@{ start = 0.5; end = 2.5; text = 'a' },
                  [pscustomobject]@{ start = 3.5; end = 6.0; text = 'b' })
        $out = Merge-Turns -Segments $segs -Turns $script:TURNS
        $out[0].speaker | Should -Be 'Speaker 1'   # SPEAKER_01 is first by time
        $out[1].speaker | Should -Be 'Speaker 2'
    }
    It 'leaves speaker null for a segment with no overlap' {
        $segs = @([pscustomobject]@{ start = 20.0; end = 25.0; text = 'x' })
        (Merge-Turns -Segments $segs -Turns $script:TURNS)[0].speaker | Should -BeNullOrEmpty
    }
}

Describe 'Format-DiarizedTranscript' {
    BeforeAll {
        $script:DSEGS = @(
            [pscustomobject]@{ start = 0.0; end = 1.0; text = 'hello'; speaker = 'Speaker 1' },
            [pscustomobject]@{ start = 1.0; end = 2.5; text = 'world'; speaker = 'Speaker 2' })
    }
    It 'txt prefixes each turn with a [MM:SS] clock + speaker (parity with meta.timestamps=diarize)' {
        $t = Format-DiarizedTranscript -Segments $script:DSEGS -Fmt 'txt'
        $t | Should -Match '\[00:00\] Speaker 1: hello'
        $t | Should -Match '\[00:01\] Speaker 2: world'
    }
    It 'html (dashboard) omits the clock (parity with to_text timestamps=False)' {
        $t = Format-DiarizedTranscript -Segments $script:DSEGS -Fmt 'html'
        $t | Should -Match 'Speaker 1: hello'
        $t | Should -Not -Match '\[00:00\]'
    }
    It 'srt emits indexed cues with SRT timestamps + speaker' {
        $t = Format-DiarizedTranscript -Segments $script:DSEGS -Fmt 'srt'
        $t | Should -Match '00:00:00,000 --> 00:00:01,000'
        $t | Should -Match 'Speaker 1: hello'
    }
    It 'vtt uses the voice tag with VTT timestamps' {
        $t = Format-DiarizedTranscript -Segments $script:DSEGS -Fmt 'vtt'
        $t | Should -Match 'WEBVTT'
        $t | Should -Match '00:00:01\.000 --> 00:00:02\.500'
        $t | Should -Match '<v Speaker 1>hello'
    }
    It 'json carries the speaker field per segment' {
        $t = Format-DiarizedTranscript -Segments $script:DSEGS -Fmt 'json'
        ($t | ConvertFrom-Json).segments[0].speaker | Should -Be 'Speaker 1'
    }
    It 'json keeps segments as an array even for a single segment' {
        $one = @([pscustomobject]@{ start = 0.0; end = 1.0; text = 'hi'; speaker = 'Speaker 1' })
        $t = Format-DiarizedTranscript -Segments $one -Fmt 'json'
        # Must serialize as a list, not a bare object (5.1 single-element unwrap).
        $t | Should -Match '"segments":\s*\['
        @(($t | ConvertFrom-Json).segments).Count | Should -Be 1
    }
}
