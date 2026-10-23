# RAG Regression Lab demo: compare two val runs and open the HTML report.
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1          # reuse stored runs
#   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Fresh   # re-run both configs first (~50 min)
param([switch]$Fresh)

Set-Location (Split-Path -Parent $PSScriptRoot)    # repo root

$Baseline  = "val-5352aebf-20261017-031708"
$Candidate = "val-11a459fe-b9rerun"

if ($Fresh) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Baseline  = "val-demo-baseline-$stamp"
    $Candidate = "val-demo-chunk1000-$stamp"
    python -m ragregress.cli run --config configs/baseline.yaml --run-id $Baseline
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -m ragregress.cli run --config configs/chunk_1000.yaml --run-id $Candidate
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# compare exits 0 / 6 / 7 for PASS / FAIL / INCONCLUSIVE; any other code is an error
python -m ragregress.cli compare --baseline $Baseline --candidate $Candidate --worst 5
if ($LASTEXITCODE -notin 0, 6, 7) { exit $LASTEXITCODE }

python -m ragregress.cli report --baseline $Baseline --candidate $Candidate --targets configs/targets/squad_val.yaml --open
exit $LASTEXITCODE