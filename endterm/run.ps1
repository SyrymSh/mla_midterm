param(
    [ValidateSet('Validate','Train')][string]$Stage='Validate',
    [string]$PythonExecutable=''
)
$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
if (-not $PythonExecutable) {
    $localPython=Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localPython) {$PythonExecutable=$localPython}
    else {
        $pythonCommand=Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) {throw 'Create .venv or pass -PythonExecutable with the path to your Python interpreter.'}
        $PythonExecutable=$pythonCommand.Source
    }
}
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
Push-Location $projectRoot
try {
    if ($Stage -eq 'Validate') {
        & $PythonExecutable -m unittest discover -s tests -v
        if ($LASTEXITCODE -ne 0) {throw 'Data integrity tests failed'}
        foreach ($script in @('audit_data.py','validate_submission.py')) {
            & $PythonExecutable (Join-Path $projectRoot "src\$script")
            if ($LASTEXITCODE -ne 0) {throw "Failed: $script"}
        }
    } elseif ($Stage -eq 'Train') {
        foreach ($script in @('prepare.py','audit_data.py','train.py','cluster.py','plots.py')) {
            & $PythonExecutable (Join-Path $projectRoot "src\$script")
            if ($LASTEXITCODE -ne 0) {throw "Failed: $script"}
        }
    }
    if ($LASTEXITCODE -ne 0) {throw "Stage failed: $Stage"}
} finally {Pop-Location}
