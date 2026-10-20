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
    PerPano x panos. The sum is ALL-TIME for the org, not this month's: it fails safe (it
    can only stop early), so after a credit reset raise -Cap by what earlier months spent.
    A split in progress has not written its row yet, so several streams sharing one org
    can each overshoot by at most one split; size the cap for that.

    It also stops when compare.py exits non-zero: exit 3 means a leg was not runnable or
    aborted after 10 consecutive failures (an outage, or the org's credits running out),
    and the next split would almost certainly do the same. Isolated per-pano failures
    exit 0 and do not stop it; compare.py prints them and a re-run fills them.

    Runs under Windows PowerShell 5.1 and PowerShell 7. API keys are read from Windows
    USER environment variables, never from a file:
    ANTHROPIC_API_KEY_CLAUDE5, ANTHROPIC_API_KEY_CLAUDE20, ANTHROPIC_API_KEY_TEAM.

.PARAMETER DataRoot
    The checkout that holds benchmark/*/panos, .model_cache and the ledger. Defaults to the
    MAIN checkout (git rev-parse --git-common-dir), even when this script is run from a
    worktree: panos and the cache are gitignored, so a worktree has neither, and a ledger
    written into a worktree is deleted with it (#139, #143).

.PARAMETER Splits
    One or more split names. `-Splits annapolis,bend` and `-Splits 'annapolis,bend'` both
    work; the second is what `powershell -File` hands over.

.PARAMETER Python
    The interpreter that runs compare.py. Defaults to the main checkout's
    .venv\Scripts\python.exe when it exists (the Windows dev venv, which has the anthropic
    SDK), else `python` from PATH.

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
  [string]$DataRoot,
  [string]$Python
)
$ErrorActionPreference = 'Stop'
$code = (Resolve-Path "$PSScriptRoot\..\..").Path

# The main checkout, the same answer rampnet/ledger.py's canonical_repo_root gives: git
# prints a bare ".git" from the main checkout and an absolute path from a worktree. Falls
# back to this script's checkout when git cannot answer (no git, not a clone).
function Get-MainCheckout {
  # Function-local: under 5.1, 'Stop' makes even a 2>$null-redirected git error fatal.
  $ErrorActionPreference = 'Continue'
  try { $common = & git -C $code rev-parse --git-common-dir 2>$null } catch { return $code }
  if ($LASTEXITCODE -ne 0 -or -not $common) { return $code }
  if (-not [IO.Path]::IsPathRooted($common)) { $common = Join-Path $code $common }
  return (Split-Path -Parent (Resolve-Path $common).Path)
}
if (-not $DataRoot) { $DataRoot = Get-MainCheckout }
if (-not $Python) {
  $venv = Join-Path $DataRoot '.venv\Scripts\python.exe'
  $Python = if (Test-Path $venv) { $venv } else { 'python' }
}
# `powershell -File ... -Splits a,b` binds the single string 'a,b'; split it here.
$Splits = @($Splits | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } |
    Where-Object { $_ })

$ledger = Join-Path $DataRoot 'analysis_out\usage_log.jsonl'
$keyVar = @{ claude5 = 'ANTHROPIC_API_KEY_CLAUDE5'; claude20 = 'ANTHROPIC_API_KEY_CLAUDE20'; 'claude-team' = 'ANTHROPIC_API_KEY_TEAM' }[$Account]
$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable($keyVar, 'User')
if (-not $env:ANTHROPIC_API_KEY) { throw "no key in user env var $keyVar" }
"code $code; data, cache and ledger in $DataRoot; python $Python"

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
  # 'Continue' around the native call only. Under Windows PowerShell 5.1, 'Stop' plus
  # 2>&1 turns compare.py's first stderr line into a terminating error that tears the
  # pipeline down mid-leg, so its `finally: report_usage` never runs and calls already
  # paid for get no ledger row (the #139 failure mode). PowerShell 7.2+ no longer does
  # this, but the script is not pinned to 7. The exit status is read below instead.
  $ErrorActionPreference = 'Continue'
  try {
    # --usage-log names the same file the cap reads; compare.py's own default is the code
    # checkout's canonical root, which is a different file whenever -DataRoot points elsewhere.
    & $Python (Join-Path $code 'scripts\model_comparison\compare.py') (Join-Path $DataRoot "benchmark\$s") `
        --models "claude:$Model" --claude-serving-path anthropic --claude-effort low `
        --api-account $Account --cache-dir (Join-Path $DataRoot '.model_cache') `
        --usage-log $ledger 2>&1 |
      ForEach-Object { "$_" } |
      Select-String -Pattern 'usage logged|ledger now|failure|Error|WARNING|not runnable|INCOMPLETE|^claude-' |
      ForEach-Object { "  $($_.Line)" }
    $rc = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = 'Stop'
  }
  if ($rc -ne 0) { "STOP: compare.py exited $rc on $s"; break }
}
"{0} [{1}] done; ledger total for {1}: `${2:N2}" -f (Get-Stamp), $Account, (Get-AccountSpend)
