[CmdletBinding()]
param(
    [string]$PythonInterpreter,
    [string]$EnvironmentName = 'pfmval_py310'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

function Resolve-ExplicitPythonInterpreter {
    param([Parameter(Mandatory = $true)][string]$Value)

    if (Test-Path -LiteralPath $Value -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Value).Path
    }

    $command = Get-Command -Name $Value -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    throw "指定的 -PythonInterpreter 不存在或不可执行: $Value"
}

function Get-CondaRegisteredEnvironmentPrefixes {
    $conda = Get-Command -Name conda -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $conda) {
        try {
            $raw = @(& $conda.Name env list --json 2>$null)
            if ($LASTEXITCODE -eq 0) {
                $envList = ($raw -join "`n") | ConvertFrom-Json
                if ($null -ne $envList.envs) {
                    return @($envList.envs | ForEach-Object { [string]$_ })
                }
            }
        }
        catch {
            # Conda CLI 不可用时，继续读取其用户级已登记环境清单。
        }
    }

    $registry = Join-Path $env:USERPROFILE '.conda\environments.txt'
    if (Test-Path -LiteralPath $registry -PathType Leaf) {
        return @(Get-Content -LiteralPath $registry | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    }

    return @()
}

if (-not [string]::IsNullOrWhiteSpace($PythonInterpreter)) {
    Write-Output (Resolve-ExplicitPythonInterpreter -Value $PythonInterpreter)
    return
}

foreach ($prefix in Get-CondaRegisteredEnvironmentPrefixes) {
    $normalizedPrefix = $prefix.TrimEnd('\', '/')
    if ([string]::Equals((Split-Path -Path $normalizedPrefix -Leaf), $EnvironmentName, [System.StringComparison]::OrdinalIgnoreCase)) {
        $candidate = Join-Path $normalizedPrefix 'python.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            Write-Output (Resolve-Path -LiteralPath $candidate).Path
            return
        }
    }
}

throw "无法从本机 Conda 已登记环境中定位 '$EnvironmentName'。请显式提供 -PythonInterpreter <本机 python.exe 路径>。"
