$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$localPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pathFile = Join-Path $projectRoot "config\local_python_path.txt"

if (Test-Path -LiteralPath $localPython) {
    $pythonPath = $localPython
} elseif (Test-Path -LiteralPath $pathFile) {
    $pythonPath = (Get-Content -Raw -Encoding UTF8 -LiteralPath $pathFile).Trim()
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python not found. Create .venv or put a Python executable path in config/local_python_path.txt."
    }
    $pythonPath = $pythonCommand.Source
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python executable not found: $pythonPath"
}

& $pythonPath -m streamlit run (Join-Path $projectRoot "streamlit_app.py") --server.address 127.0.0.1 --server.port 8501
