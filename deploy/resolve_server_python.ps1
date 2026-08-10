# resolve_server_python.ps1
# Single source of truth for the Windows server Python interpreter.
# Do not fall back to PATH: WindowsApps/python.exe is only a Store alias.

function Get-PfmvalServerPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathId,
        [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
    )

    $profilePath = Join-Path $ProjectRoot "configs\server_paths.yaml"
    if (-not (Test-Path -LiteralPath $profilePath -PathType Leaf)) {
        throw "Server machine profile not found: $profilePath"
    }
    $inPaths = $false
    $inTarget = $false
    $candidate = ""
    foreach ($line in Get-Content -LiteralPath $profilePath) {
        if ($line -match '^paths:\s*$') {
            $inPaths = $true
            continue
        }
        if ($inPaths -and $line -match '^\S') {
            break
        }
        if ($inPaths -and $line -match '^\s{2}(?<id>[^:]+):\s*$') {
            $inTarget = $Matches.id.Trim() -eq $PathId
            continue
        }
        if ($inTarget -and $line -match '^  \S') {
            break
        }
        if ($inTarget -and $line -match '^    path:\s*["''](?<path>.+?)["'']\s*$') {
            $candidate = $Matches.path.Trim()
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        throw "Registered server path is missing: $PathId"
    }
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        throw "Registered server path must be absolute: $PathId -> $candidate"
    }
    return $candidate
}

function Resolve-PfmvalServerPython {
    [CmdletBinding()]
    param(
        [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
    )

    if (-not [string]::IsNullOrWhiteSpace($env:PFMVAL_CONFIG) -or -not [string]::IsNullOrWhiteSpace($env:PFMVAL_PYTHON)) {
        throw "PFMVAL_CONFIG and PFMVAL_PYTHON are retired; clear them and use configs\\server_paths.yaml"
    }

    $profilePath = Join-Path $ProjectRoot "configs\server_paths.yaml"
    if (-not (Test-Path -LiteralPath $profilePath -PathType Leaf)) {
        throw "Server machine profile not found: $profilePath"
    }
    $inRuntime = $false
    $candidate = ""
    foreach ($line in Get-Content -LiteralPath $profilePath) {
        if ($line -match '^runtime:\s*$') {
            $inRuntime = $true
            continue
        }
        if ($inRuntime -and $line -match '^\S') {
            break
        }
        if ($inRuntime -and $line -match '^\s+python_interpreter:\s*["''](?<path>.+?)["'']\s*$') {
            $candidate = $Matches.path.Trim()
            break
        }
    }

    if ([string]::IsNullOrWhiteSpace($candidate)) {
        throw "runtime.python_interpreter is missing from server_paths.yaml"
    }
    if (-not [System.IO.Path]::IsPathRooted($candidate)) {
        throw "Python interpreter must be an absolute path: $candidate"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Server Python interpreter not found: $candidate"
    }

    $resolved = (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
    if ([System.IO.Path]::GetExtension($resolved).ToLowerInvariant() -ne ".exe") {
        throw "Python interpreter must be an .exe file: $resolved"
    }
    if ($resolved -match "(?i)\\WindowsApps\\python(?:\.exe)?$") {
        throw "WindowsApps Python alias is forbidden"
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
