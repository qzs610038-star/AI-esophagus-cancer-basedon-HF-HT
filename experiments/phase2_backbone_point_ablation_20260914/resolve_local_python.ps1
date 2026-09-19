[CmdletBinding()]
param(
    [string]$PythonInterpreter,
    [string]$EnvironmentName = 'pfmval_py310'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

if (-not [string]::IsNullOrWhiteSpace($PythonInterpreter)) {
    if (Test-Path -LiteralPath $PythonInterpreter -PathType Leaf) {
        Write-Output (Resolve-Path -LiteralPath $PythonInterpreter).Path
        return
    }
    $explicit = Get-Command -Name $PythonInterpreter -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $explicit) { Write-Output $explicit.Source; return }
    throw "指定的本地 Python 不存在: $PythonInterpreter"
}

$prefixes = @()
$conda = Get-Command -Name conda -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $conda) {
    try {
        $raw = @(& $conda.Name env list --json 2>$null)
        if ($LASTEXITCODE -eq 0) { $prefixes = @(($raw -join "`n" | ConvertFrom-Json).envs) }
    } catch { $prefixes = @() }
}
if ($prefixes.Count -eq 0) {
    $registry = Join-Path $env:USERPROFILE '.conda\environments.txt'
    if (Test-Path -LiteralPath $registry -PathType Leaf) {
        $prefixes = @(Get-Content -LiteralPath $registry | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    }
}
foreach ($prefix in $prefixes) {
    $normalized = ([string]$prefix).TrimEnd('\', '/')
    if ([string]::Equals((Split-Path $normalized -Leaf), $EnvironmentName, [System.StringComparison]::OrdinalIgnoreCase)) {
        $candidate = Join-Path $normalized 'python.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            Write-Output (Resolve-Path -LiteralPath $candidate).Path
            return
        }
    }
}
throw "无法定位本地 Conda 环境 '$EnvironmentName'；请显式传入 -PythonInterpreter。"
