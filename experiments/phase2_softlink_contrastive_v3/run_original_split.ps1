<#
Run the original six-patient spatial-block split only.  This wrapper never
dispatches a leave-one-patient-out protocol.
#>
param(
    [string]$Config = (Join-Path $PSScriptRoot 'config.json'),
    [int[]]$Seeds = @(42),
    [switch]$Resume,
    [string]$ResumeFrom,
    [switch]$PlanOnly,
    [string]$Batch,
    [string]$PythonInterpreter,
    [string]$RunsRoot,
    [string]$WeightsRoot
)

$ErrorActionPreference = 'Stop'
$arguments = @{ Protocol = 'original'; Config = $Config; Seeds = $Seeds }
if ($Resume) { $arguments.Resume = $true }
if ($PSBoundParameters.ContainsKey('ResumeFrom')) { $arguments.ResumeFrom = $ResumeFrom }
if ($PlanOnly) { $arguments.PlanOnly = $true }
if ($PSBoundParameters.ContainsKey('Batch')) { $arguments.Batch = $Batch }
if ($PSBoundParameters.ContainsKey('PythonInterpreter')) { $arguments.PythonInterpreter = $PythonInterpreter }
if ($PSBoundParameters.ContainsKey('RunsRoot')) { $arguments.RunsRoot = $RunsRoot }
if ($PSBoundParameters.ContainsKey('WeightsRoot')) { $arguments.WeightsRoot = $WeightsRoot }
& (Join-Path $PSScriptRoot 'run.ps1') @arguments
exit $LASTEXITCODE
