<#
Run the complete six-patient leave-one-patient-out protocol.  The defaults
cover all six folds and the three planned seeds; use -Folds/-Seeds for a
deliberate scoped continuation.
#>
param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [string[]]$Folds = @('HYZ15040', 'JFX', 'LMZ12939', 'TGC', 'XSL', 'ZHZ'),
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$Resume,
    [string]$ResumeFrom,
    [switch]$PlanOnly,
    [string]$Batch,
    [string]$PythonInterpreter,
    [string]$RunsRoot,
    [string]$WeightsRoot
)

$ErrorActionPreference = 'Stop'
$arguments = @{ Protocol = 'lopo6'; Config = $Config; Folds = $Folds; Seeds = $Seeds }
if ($Resume) { $arguments.Resume = $true }
if ($PSBoundParameters.ContainsKey('ResumeFrom')) { $arguments.ResumeFrom = $ResumeFrom }
if ($PlanOnly) { $arguments.PlanOnly = $true }
if ($PSBoundParameters.ContainsKey('Batch')) { $arguments.Batch = $Batch }
if ($PSBoundParameters.ContainsKey('PythonInterpreter')) { $arguments.PythonInterpreter = $PythonInterpreter }
if ($PSBoundParameters.ContainsKey('RunsRoot')) { $arguments.RunsRoot = $RunsRoot }
if ($PSBoundParameters.ContainsKey('WeightsRoot')) { $arguments.WeightsRoot = $WeightsRoot }
& (Join-Path $PSScriptRoot 'run.ps1') @arguments
exit $LASTEXITCODE
