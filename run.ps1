<#
.SYNOPSIS
  Task runner for LineSight on Windows. Everything the Makefile does, without
  needing make.

.DESCRIPTION
  There is no `make` on a stock Windows box, PowerShell has no equivalent, and
  the Makefile's $(HOME) resolves to the wrong place under WSL (which sees a
  Linux home, not C:\Users\<you>, and cannot execute the Windows venv anyway).
  So this script is the supported path on Windows and the Makefile is for
  Linux, macOS and CI.

  The venv lives OUTSIDE the repo at ~\.venvs\linesight. That is not a style
  choice: this project sits ~90 characters deep, Windows caps paths at 260 by
  default, and torch ships licence files with ~160-character relative paths, so
  an in-repo .venv fails to install torch with WinError 206.

.EXAMPLE
  .\run.ps1 help
  .\run.ps1 install
  .\run.ps1 test
  .\run.ps1 fit -Sku aitex_02 -- --normal data/aitex/normal_02
  .\run.ps1 run -Sku aitex_02 -- --source data/aitex/roll_02
  .\run.ps1 mark -Fabric 02  # the standalone defect marker
  .\run.ps1 learn            # bench rig: watch clean cloth, then `inspect`
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Task = "help",

    # mark / probe options
    [string]$Fabric = "02",
    [int]$N = 3,
    [string]$Probe = "p09_mark_defects",

    # SKU whose config the fit / calibrate / run tasks load
    [string]$Sku = "synthetic",

    # everything after -- is forwarded to the underlying command
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# --------------------------------------------------------------------------- #
# venv discovery
# --------------------------------------------------------------------------- #

$VenvRoot = Join-Path $HOME ".venvs\linesight"
$Py = Join-Path $VenvRoot "Scripts\python.exe"

# timm pulls the frozen ImageNet weights from the HF hub, which prints an
# unauthenticated-request notice on every single run. The weights are cached
# after the first fetch and the notice says nothing actionable, so quiet it --
# a run should not open with a warning that says nothing actionable.
if (-not $env:HF_HUB_VERBOSITY) { $env:HF_HUB_VERBOSITY = "error" }

function Assert-Venv {
    if (-not (Test-Path $Py)) {
        Write-Host "no virtualenv at $VenvRoot" -ForegroundColor Yellow
        Write-Host "run:  .\run.ps1 install" -ForegroundColor Yellow
        exit 1
    }
}

function Invoke-Py {
    param([string[]]$Arguments)
    Assert-Venv
    # Windows PowerShell 5.1 turns any stderr output from a native executable
    # into a NativeCommandError and, with ErrorActionPreference=Stop, kills the
    # script -- even when the process exited 0. torch and huggingface_hub both
    # write harmless notices to stderr, so a successful run would abort here.
    # The exit code is the only trustworthy signal, so gate on that alone.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Py @Arguments
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #

function Task-Help {
    Write-Host ""
    Write-Host "LineSight tasks" -ForegroundColor Cyan
    Write-Host "  .\run.ps1 <task>"
    Write-Host ""
    Write-Host "setup" -ForegroundColor Cyan
    Write-Host "  install       create the venv and install linesight (editable)"
    Write-Host "  fetch         download AITEX + Fabric Stain (needs ~/.kaggle/kaggle.json)"
    Write-Host ""
    Write-Host "quality" -ForegroundColor Cyan
    Write-Host "  test          run the full suite"
    Write-Host "  test-fast     skip anything marked slow or data"
    Write-Host "  lint          ruff check"
    Write-Host "  fmt           ruff format + autofix"
    Write-Host ""
    Write-Host "bench rig" -ForegroundColor Cyan
    Write-Host "  tape          print the ArUco position tape (PDF)"
    Write-Host "  check         camera bring-up: fps, focus, exposure, tape"
    Write-Host "  align         live view with the ROI and tape drawn on it"
    Write-Host "  learn         watch clean fabric: fit the bank and the threshold"
    Write-Host "  inspect       score the sheet now under the camera"
    Write-Host ""
    Write-Host "offline" -ForegroundColor Cyan
    Write-Host "  fit           -Sku <name>   build a memory bank from a folder"
    Write-Host "  calibrate     -Sku <name>   derive a threshold from a budget"
    Write-Host "  run           -Sku <name>   inspect a roll, print the table"
    Write-Host "  whitepaper    rebuild the whitepaper PDF from results/"
    Write-Host ""
    Write-Host "verification" -ForegroundColor Cyan
    Write-Host "  mark          -Fabric 02 -N 3   mark defects on real AITEX images"
    Write-Host "  probe         -Probe p04_tiling  run one probe"
    Write-Host ""
    Write-Host "  clean         remove caches"
    Write-Host ""
}

function Task-Install {
    if (-not (Test-Path $Py)) {
        Write-Host "creating venv at $VenvRoot ..." -ForegroundColor Cyan
        python -m venv $VenvRoot
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -e ".[api,report,dev]" `
        --extra-index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "`ninstalled. next:  .\run.ps1 test" -ForegroundColor Green
}

function Task-Fetch {
    Assert-Venv
    $kaggle = Join-Path $HOME ".kaggle\kaggle.json"
    if (-not (Test-Path $kaggle)) {
        Write-Host "no Kaggle token at $kaggle" -ForegroundColor Yellow
        Write-Host "Kaggle -> Settings -> API -> Create New Token" -ForegroundColor Yellow
        exit 1
    }
    & $Py -m pip install -q kaggle
    New-Item -ItemType Directory -Force data | Out-Null
    $cli = Join-Path $VenvRoot "Scripts\kaggle.exe"
    & $cli datasets download -d nexuswho/aitex-fabric-image-database -p data/aitex --unzip
    & $cli datasets download -d priemshpathirana/fabric-stain-dataset -p data/fabric_stain --unzip
    Write-Host "local datasets ready (MVTec is deliberately not fetched -- ADR-003)" -ForegroundColor Green
}

function Task-Clean {
    foreach ($d in @(".pytest_cache", ".ruff_cache")) {
        if (Test-Path $d) { Remove-Item -Recurse -Force $d }
    }
    Get-ChildItem -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
    Write-Host "cleaned" -ForegroundColor Green
}

# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

switch ($Task.ToLower()) {
    "help"       { Task-Help }
    "install"    { Task-Install }
    "fetch"      { Task-Fetch }

    "test"       { Invoke-Py (@("-m", "pytest") + $Rest) }
    "test-fast"  { Invoke-Py (@("-m", "pytest", "-m", "not slow and not data") + $Rest) }
    "lint"       { Invoke-Py @("-m", "ruff", "check", "src", "tests", "probes", "tools") }
    "fmt"        {
        Invoke-Py @("-m", "ruff", "format", "src", "tests", "probes", "tools")
        Invoke-Py @("-m", "ruff", "check", "--fix", "src", "tests", "probes", "tools")
    }

    "fit"        { Invoke-Py (@("-m", "linesight", "fit", "--sku", $Sku) + $Rest) }
    "calibrate"  { Invoke-Py (@("-m", "linesight", "calibrate", "--sku", $Sku) + $Rest) }
    "run"        { Invoke-Py (@("-m", "linesight", "run", "--sku", $Sku) + $Rest) }


    "mark"       {
        Invoke-Py (@("probes/p09_mark_defects.py", "--fabric", $Fabric,
                     "--n", "$N") + $Rest)
    }
    "probe"      { Invoke-Py (@("probes/$Probe.py") + $Rest) }

    "tape"       { Invoke-Py (@("tools/make_tape.py") + $Rest) }
    "check"      { Invoke-Py (@("probes/p13_phone_stream.py") + $Rest) }
    "align"      { Invoke-Py (@("tools/align_view.py") + $Rest) }
    "learn"      { Invoke-Py (@("tools/bench_run.py", "learn") + $Rest) }
    "inspect"    { Invoke-Py (@("tools/bench_run.py", "inspect") + $Rest) }
    "whitepaper" { Invoke-Py (@("tools/make_whitepaper.py") + $Rest) }

    "clean"      { Task-Clean }

    default {
        Write-Host "unknown task '$Task'" -ForegroundColor Red
        Task-Help
        exit 1
    }
}
