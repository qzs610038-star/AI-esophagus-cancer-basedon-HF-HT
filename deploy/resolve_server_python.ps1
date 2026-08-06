# resolve_server_python.ps1
# Single source of truth for the Windows server Python interpreter.
# Do not fall back to PATH: WindowsApps/python.exe is only a Store alias.

function Resolve-PfmvalServerPython {
    [CmdletBinding()]
    param()

    $defaultPath = "C:\Users\AIPatho1\pfmval_env\Scripts\python.exe"
    $candidate = if ($env:PFMVAL_PYTHON) {
        $env:PFMVAL_PYTHON.Trim()
    } else {
        $defaultPath
    }

    if ([string]::IsNullOrWhiteSpace($candidate)) {
        throw "PFMVAL_PYTHON is empty; refusing PATH-based Python lookup"
    }
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        throw "Python interpreter must be an absolute path: $candidate"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Server Python interpreter not found: $candidate (expected $defaultPath or an explicit PFMVAL_PYTHON override)"
    }

    $resolved = (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
    if ([System.IO.Path]::GetExtension($resolved).ToLowerInvariant() -ne ".exe") {
        throw "Python interpreter must be an .exe file: $resolved"
    }
    if ($resolved -match "(?i)\\WindowsApps\\python(?:\.exe)?$") {
        throw "WindowsApps Python alias is forbidden; use $defaultPath"
    }
    return $resolved
}

function Get-PfmvalPythonIdentity {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath
    )

    $versionOutput = @(& $PythonPath --version 2>&1)
    $versionExit = $LASTEXITCODE
    $version = (($versionOutput | ForEach-Object { "$_" }) -join " ").Trim()
    if ($versionExit -ne 0 -or [string]::IsNullOrWhiteSpace($version)) {
        throw "Python version probe failed (exit=$versionExit): $version"
    }

    $exeOutput = @(& $PythonPath -c "import sys; print(sys.executable)" 2>&1)
    $exeExit = $LASTEXITCODE
    $actual = (($exeOutput | ForEach-Object { "$_" }) | Select-Object -Last 1)
    $actual = if ($null -eq $actual) { "" } else { "$actual".Trim() }
    if ($exeExit -ne 0 -or [string]::IsNullOrWhiteSpace($actual)) {
        throw "Python executable identity probe failed (exit=$exeExit): $actual"
    }

    $expectedFull = [System.IO.Path]::GetFullPath($PythonPath)
    $actualFull = [System.IO.Path]::GetFullPath($actual)
    if ($actualFull -ine $expectedFull) {
        throw "Python executable mismatch: requested=$expectedFull actual=$actualFull"
    }

    return [pscustomobject]@{
        Path       = $expectedFull
        Version    = $version
        Executable = $actualFull
    }
}
