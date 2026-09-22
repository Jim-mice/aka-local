# AKA-Local Lab CLI (Phase 11)
# Multi-operator CUDA optimization framework
param(
    [Parameter(Position=0)]
    [string]$Command = "status",

    [string]$Environment = "v100",
    [string]$Operator = "rms_norm_v100_cuda",
    [string]$OpAlias = "",
    [int]$Episodes = 1,
    [string]$Candidate = "",
    [string]$EnvAlias = "",
    [int]$EpisodeNum = 0,
    [switch]$NoAgent = $false,
    [string]$Interpreter = ""
)

$ROOT = "$PSScriptRoot"
$LAUNCHER = "$ROOT\scripts\run_lab.py"
if ($Interpreter) {
    $PYTHON = $Interpreter
} else {
    $PYTHON = "$ROOT\.venv\Scripts\python.exe"
    if (-not (Test-Path $PYTHON)) {
        throw "D-repo virtual environment is missing: $PYTHON. Pass -Interpreter <path-to-python>; no global editable-install fallback is allowed."
    }
}
if (-not (Test-Path $PYTHON)) {
    throw "Python interpreter not found: $PYTHON"
}
if (-not (Test-Path $LAUNCHER)) {
    throw "D-repo launcher missing: $LAUNCHER"
}
$ENV:PATH = "$(Split-Path $PYTHON -Parent);$ENV:PATH"

# Resolve parameter aliases
if ($OpAlias) { $Operator = $OpAlias }
if ($EnvAlias) { $Environment = $EnvAlias }

# Map short names
if ($Operator -eq "rms_norm") { $Operator = "rms_norm_v100_cuda" }
if ($Operator -eq "layer_norm") { $Operator = "layer_norm_v100_cuda" }

function header { Write-Host "`n=== AKA-Lab: $args ===" -ForegroundColor Cyan }

switch ($Command) {
    "status" {
        & $PYTHON $LAUNCHER --module lab.cli status
    }

    "campaigns" {
        & $PYTHON $LAUNCHER --module lab.cli campaigns
    }

    "run" {
        $noAgentFlag = if ($NoAgent) { "--no-agent" } else { "" }
        header "Run Campaign: env=$Environment op=$Operator episodes=$Episodes no_agent=$NoAgent"
        & $PYTHON $LAUNCHER --module lab.cli run --env $Environment --op $Operator --episodes $Episodes $noAgentFlag
    }

    "evaluate" {
        if (-not $Candidate) {
            Write-Host "Usage: lab evaluate --candidate path/to/candidate.cu [--env v100] [--op NAME]" -ForegroundColor Yellow
            return
        }
        header "Evaluate: candidate=$Candidate op=$Operator"
        & $PYTHON $LAUNCHER --module lab.cli evaluate --candidate $Candidate --env $Environment --op $Operator
    }

    "knowledge" {
        & $PYTHON $LAUNCHER --module lab.cli knowledge
    }

    "recover" {
        & $PYTHON $LAUNCHER --module lab.cli recover
    }

    "doctor" {
        & $PYTHON $LAUNCHER --module lab.cli doctor --env $Environment
    }

    "replay" {
        $epNum = if ($EpisodeNum) { $EpisodeNum } else { $Episodes }
        if ($epNum -eq 0) {
            Write-Host "Usage: lab replay --ep N [--op NAME]" -ForegroundColor Yellow
            return
        }
        header "Replay Episode $epNum"
        & $PYTHON $LAUNCHER --module lab.cli replay --env $Environment --op $Operator --ep $epNum
    }

    "list-ops" {
        & $PYTHON $LAUNCHER --module lab.cli list-ops
    }

    "validate" {
        & $PYTHON $LAUNCHER --module lab.cli validate
    }

    default {
        Write-Host @"

AKA-Lab CLI (Phase 11) - Multi-operator CUDA Optimization Framework

Commands:
  lab status                     Show project status
  lab campaigns                  List all campaigns
  lab run [--env v100] [--op NAME] [--episodes N] [--NoAgent]
  lab evaluate --candidate PATH [--env v100] [--op NAME]
  lab knowledge                  Show knowledge
  lab recover                    Validate and repair state
  lab doctor [--env v100]        Run self-test
  lab replay --ep N [--op NAME]  Replay episode
  lab list-ops                   List available operators
  lab validate                   Run validation suite

Examples:
  lab run --env v100 --op rms_norm_v100_cuda --episodes 3
  lab run --env v100 --op layer_norm_v100_cuda --episodes 1 --NoAgent
  lab evaluate --candidate operators/layer_norm_v100_cuda/reference.cu --op layer_norm_v100_cuda
  lab list-ops
  lab doctor --env v100
"@
    }
}
