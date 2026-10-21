# Stage 2 batch launcher (2026-10-15). Runs one strategy on one provider and writes a uniquely named results file.
# Keys are loaded from Windows USER environment variables and never printed.
#   powershell -File stage2_batch.ps1 -Provider anthropic -Strategy multi_agent -Scenarios all -Seeds 2 -Tag s2
param(
    [Parameter(Mandatory)][ValidateSet("anthropic", "openai")][string]$Provider,
    [Parameter(Mandatory)][string]$Strategy,
    [Parameter(Mandatory)][string]$Scenarios,
    [Parameter(Mandatory)][string]$Seeds,
    [Parameter(Mandatory)][string]$Tag
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
$env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', 'User')
$env:PYTHONIOENCODING = "utf-8"

$short = if ($Provider -eq "anthropic") { "claude" } else { "gpt" }
$out = "results\stage2-$short-$Strategy-$Tag.jsonl"
if (Test-Path $out) { throw "refusing to overwrite $out" }

$pyArgs = @("run_experiment.py", "--strategies", $Strategy, "--provider", $Provider,
            "--scenarios", $Scenarios, "--seeds", $Seeds, "--out", $out)
if ($Provider -eq "openai") {
    $pyArgs += @("--model", "gpt-5.5-2026-04-23", "--price-in", "5", "--price-out", "30", "--price-cache-read", "0.5")
}
$ErrorActionPreference = "Continue"
python @pyArgs
