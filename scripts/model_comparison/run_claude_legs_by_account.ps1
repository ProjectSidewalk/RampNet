<#
.SYNOPSIS
    Run one Claude leg over a list of splits, paid by one first-party API org, and stop
    before a split whose projected cost would push that org past a budget cap.

.DESCRIPTION
    Written for the newest-ids campaign (docs/claude_newest_ids.md, 2026-10-19), which paid
    for Opus 5.5 and Fable 5.1 legs out of three subscriptions' monthly API credits. Each
    subscription's credit sits in its own Console org with its own API key and expiry, so
    each org runs its own invocation of this script, and --api-account tags every ledger
    row with the org that paid.

    The cap is checked against analysis_out/usage_log.jsonl (list-price est_cost_usd,
    summed over rows with this api_account) BEFORE each split, with 10% headroom on
    PerPano x panos. A split in progress has not written its row yet, so several streams
    sharing one org can each overshoot by at most one split; size the cap for that.

    API keys are read from Windows USER environment variables, never from a file:
    ANTHROPIC_API_KEY_CLAUDE5, ANTHROPIC_API_KEY_CLAUDE20, ANTHROPIC_API_KEY_TEAM.

.PARAMETER DataRoot
    The checkout that holds benchmark/*/panos, .model_cache and the ledger -- the MAIN
    checkout. Panos and the cache are gitignored, so a worktree has neither.

.EXAMPLE
    # Opus 5.5 on two splits, paid by the claude5 org, stopping before $95 of spend
    .\scripts\model_comparison\run_claude_legs_by_account.ps1 -Account claude5 `
        -Model claude-opus-5-5 -PerPano 0.0665 -Cap 95 -Splits annapolis,bend
#>
param(
  [Parameter(Mandatory)][ValidateSet('claude5', 'claude20', 'claude-team')][string]$Account,
  [Parameter(Mandatory)][string]$Model,
  [Parameter(Mandatory)][double]$PerPano,
  [Parameter(Mandatory)][double]$Cap,
  [Parameter(Mandatory)][string[]]$Splits,
  [string]$DataRoot = (Resolve-Path "$PSScriptRoot\..\..").Path,
  [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$code = (Resolve-Path "$PSScriptRoot\..\..").Path
$ledger = Join-Path $DataRoot 'analysis_out\usage_log.jsonl'
$keyVar = @{ claude5 = 'ANTHROPIC_API_KEY_CLAUDE5'; claude20 = 'ANTHROPIC_API_KEY_CLAUDE20'; 'claude-team' = 'ANTHROPIC_API_KEY_TEAM' }[$Account]
$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable($keyVar, 'User')
if (-not $env:ANTHROPIC_API_KEY) { throw "no key in user env var $keyVar" }

function Get-AccountSpend {
  $sum = 0.0
  foreach ($line in Get-Content $ledger) {
    if (-not $line.Trim()) { continue }
    $r = $line | ConvertFrom-Json
    if ($r.api_account -eq $Account -and $r.est_cost_usd) { $sum += [double]$r.est_cost_usd }
  }
  return $sum
}

function Get-Stamp { (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss') + 'Z' }

foreach ($s in $Splits) {
  $n = (Get-Content (Join-Path $DataRoot "benchmark\$s\records.jsonl") | Where-Object { $_.Trim() }).Count
  $spent = Get-AccountSpend
  $next = $n * $PerPano * 1.1
  "{0} [{1}] spent so far `${2:N2}; {3} needs <= `${4:N2} (cap `${5})" -f (Get-Stamp), $Account, $spent, $s, $next, $Cap
  if ($spent + $next -gt $Cap) { "STOP: $s would exceed the `$$Cap cap for $Account"; break }
  & $Python (Join-Path $code 'scripts\model_comparison\compare.py') (Join-Path $DataRoot "benchmark\$s") `
      --models "claude:$Model" --claude-serving-path anthropic --claude-effort low `
      --api-account $Account --cache-dir (Join-Path $DataRoot '.model_cache') 2>&1 |
    Select-String -Pattern 'usage logged|ledger now|failure|Error|^claude-' | ForEach-Object { "  $_" }
}
"{0} [{1}] done; ledger total for {1}: `${2:N2}" -f (Get-Stamp), $Account, (Get-AccountSpend)
