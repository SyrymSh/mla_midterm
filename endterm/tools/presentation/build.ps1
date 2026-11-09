param(
    [Parameter(Mandatory=$true)][string]$NodeExecutable,
    [Parameter(Mandatory=$true)][string]$NodeModules,
    [Parameter(Mandatory=$true)][string]$PythonExecutable,
    [Parameter(Mandatory=$true)][string]$PresentationSkillDir
)
$ErrorActionPreference='Stop'
$env:RUNTIME_NODE_MODULES=$NodeModules
$env:RUNTIME_PYTHON=$PythonExecutable
$env:PRESENTATION_SKILL_DIR=$PresentationSkillDir
& $NodeExecutable --stack-trace-limit=3 (Join-Path $PSScriptRoot 'build_deck.mjs')
if ($LASTEXITCODE -ne 0) {throw 'Presentation build failed'}
& $PythonExecutable (Join-Path $PSScriptRoot 'build_pdf.py')
if ($LASTEXITCODE -ne 0) {throw 'Slide PDF export failed'}
& $PythonExecutable (Join-Path $PSScriptRoot 'finalize_properties.py')
if ($LASTEXITCODE -ne 0) {throw 'Document properties update failed'}
