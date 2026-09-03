$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonCommand = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py" }

& $pythonCommand -m code_rewrite_feedback_expander.multi_expert validate-config `
  --config configs/stage1_multi_expert.json

& $pythonCommand -m unittest discover -s tests -p "test_*.py" -v

$limit = if ($env:STAGE1_SMOKE_LIMIT) { $env:STAGE1_SMOKE_LIMIT } else { "3" }
& $pythonCommand -m code_rewrite_feedback_expander.multi_expert run `
  --config configs/stage1_multi_expert.json `
  --input code_rewrite_feedback_expander/data/code_real_16.jsonl `
  --output-dir outputs/stage1_multi_expert_smoke `
  --limit $limit
