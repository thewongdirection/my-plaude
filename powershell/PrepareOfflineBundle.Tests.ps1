#requires -Version 5.1
<#
    Pester tests for tools/Prepare-OfflineBundle.ps1 (offline-bundle helper).

    Mirrors tests/test_prepare_bundle.py: pure manifest logic (dot-sourced) plus
    a child-process check that the -List CLI actually emits JSON (a bug the
    dot-sourced tests cannot see, because `exit (Invoke-Main)` would swallow it).
#>

BeforeAll {
    $script:Tool = Join-Path (Split-Path $PSScriptRoot -Parent) 'tools/Prepare-OfflineBundle.ps1'
    . $script:Tool
}

Describe 'Get-BundleManifest' {
    It 'produces ffmpeg + wheels + one item per model (parity with Python)' {
        $m = Get-BundleManifest -Models @('small', 'large-v3') -Platform 'windows' -Include @() -WithDiarizeModels $false
        ($m | ForEach-Object kind) -join ',' | Should -Be 'archive,pip,hf,hf'
        $m[0].source | Should -Match '\.zip$'
        $m[1].source | Should -Match 'faster-whisper'
        $m[1].source | Should -Match 'whisper-ctranslate2'
    }
    It 'expands cuda/diarize wheels and adds the gated model repos' {
        $m = Get-BundleManifest -Models @('large-v3') -Platform 'windows' -Include @('cuda', 'diarize') -WithDiarizeModels $true
        $pip = ($m | Where-Object kind -eq 'pip').source
        $pip | Should -Match 'nvidia-cublas-cu12'
        $pip | Should -Match 'nvidia-cudnn-cu12'
        $pip | Should -Match 'pyannote.audio'
        @($m | Where-Object kind -eq 'hf-gated').Count | Should -Be 3
        @($m | Where-Object source -match 'community-1').Count | Should -BeGreaterThan 0
    }
    It 'marks macOS FFmpeg as manual (Homebrew)' {
        $m = Get-BundleManifest -Models @('tiny') -Platform 'macos' -Include @() -WithDiarizeModels $false
        $m[0].kind | Should -Be 'manual'
        $m[0].source | Should -Match 'brew'
    }
}

Describe 'Prepare-OfflineBundle -List (CLI)' {
    # Regression: `exit (Invoke-Main)` captured the success stream, swallowing the
    # -List JSON. Exercise the real CLI in a child process and parse its stdout.
    It 'emits valid JSON to stdout and exits 0' {
        $exe = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
        $out = & $exe -NoProfile -ExecutionPolicy Bypass -File $script:Tool -List -Models small 2>&1
        $LASTEXITCODE | Should -Be 0
        $parsed = ($out -join "`n") | ConvertFrom-Json
        @($parsed | Where-Object kind -eq 'archive').Count | Should -BeGreaterThan 0
    }
}
