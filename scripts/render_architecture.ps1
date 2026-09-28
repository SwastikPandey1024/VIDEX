<#
.SYNOPSIS
    Renders VIDEX PlantUML architecture diagrams to SVG (canonical) and optional PNG.

.DESCRIPTION
    Detects Java runtime environment and Graphviz, verifies tools/plantuml.jar,
    and batch-renders docs/architecture/*.puml files into docs/architecture/rendered/.

.PARAMETER Format
    Output format: 'svg' (canonical default), 'png', or 'all' (both).

.PARAMETER IncludePng
    Switch flag to generate PNG previews alongside canonical SVG diagrams.

.PARAMETER JavaPath
    Optional explicit path to java.exe.

.PARAMETER SourceDir
    Directory containing .puml source files (default: docs/architecture).

.PARAMETER OutputDir
    Output directory for rendered diagrams (default: docs/architecture/rendered).

.EXAMPLE
    .\scripts\render_architecture.ps1
    Renders all .puml files to SVG format in docs/architecture/rendered/.

.EXAMPLE
    .\scripts\render_architecture.ps1 -IncludePng
    Renders all .puml files to both SVG and PNG formats.
#>

[CmdletBinding()]
param(
    [ValidateSet("svg", "png", "all")]
    [string]$Format = "svg",

    [switch]$IncludePng,

    [string]$JavaPath = "",

    [string]$PlantUmlJar = "tools/plantuml.jar",

    [string]$SourceDir = "docs/architecture",

    [string]$OutputDir = "docs/architecture/rendered"
)

function Invoke-CliProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = $ArgumentList -join " "
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $p = [System.Diagnostics.Process]::Start($psi)
    $out = $p.StandardOutput.ReadToEnd()
    $err = $p.StandardError.ReadToEnd()
    $p.WaitForExit()

    return [PSCustomObject]@{
        ExitCode = $p.ExitCode
        Stdout   = $out.Trim()
        Stderr   = $err.Trim()
        Combined = ("$out`n$err").Trim()
    }
}

# ── 1. Locate Java ────────────────────────────────────────────────────────────

Write-Host "==> Detecting Java runtime..." -ForegroundColor Cyan

$ResolvedJava = $null

if ($JavaPath -and (Test-Path $JavaPath)) {
    $ResolvedJava = (Resolve-Path $JavaPath).Path
}

if (-not $ResolvedJava) {
    $cmdJava = Get-Command java -ErrorAction SilentlyContinue
    if ($cmdJava) {
        $ResolvedJava = $cmdJava.Source
    }
}

if (-not $ResolvedJava -and $env:JAVA_HOME -and (Test-Path "$env:JAVA_HOME\bin\java.exe")) {
    $ResolvedJava = "$env:JAVA_HOME\bin\java.exe"
}

if (-not $ResolvedJava) {
    # Scan standard Windows JDK installation locations
    $searchPaths = @(
        "C:\Program Files\Eclipse Adoptium\*\bin\java.exe",
        "C:\Program Files\Java\*\bin\java.exe",
        "C:\Program Files\Microsoft\*\bin\java.exe",
        "C:\Program Files\Amazon Corretto\*\bin\java.exe",
        "C:\Program Files\Zulu\*\bin\java.exe",
        "C:\Program Files\JetBrains\*\jbr\bin\java.exe"
    )

    foreach ($pattern in $searchPaths) {
        $candidates = Get-Item -Path $pattern -ErrorAction SilentlyContinue
        if ($candidates) {
            $ResolvedJava = $candidates[0].FullName
            break
        }
    }
}

if (-not $ResolvedJava -or -not (Test-Path $ResolvedJava)) {
    Write-Host @"
[ERROR] Java runtime not found.
PlantUML requires Java (JRE/JDK 11+).

To install Java, run:
    winget install EclipseAdoptium.Temurin.21.JDK -e

Or specify the path manually:
    .\scripts\render_architecture.ps1 -JavaPath "C:\path\to\java.exe"
"@ -ForegroundColor Red
    exit 1
}

$verCheck = Invoke-CliProcess -FilePath $ResolvedJava -ArgumentList @("-version")
$firstLine = ($verCheck.Combined -split "`r?`n")[0]
Write-Host "    Found Java: $ResolvedJava ($firstLine)" -ForegroundColor Green


# ── 2. Locate / Verify PlantUML ───────────────────────────────────────────────

Write-Host "==> Verifying PlantUML jar..." -ForegroundColor Cyan

$ResolvedJar = Join-Path $PSScriptRoot "..\$PlantUmlJar"
if (-not (Test-Path $ResolvedJar)) {
    $ResolvedJar = $PlantUmlJar
}

if (-not (Test-Path $ResolvedJar)) {
    Write-Host "    PlantUML jar not found at $ResolvedJar. Downloading latest release..." -ForegroundColor Yellow
    $toolsDir = Split-Path -Parent $ResolvedJar
    if (-not (Test-Path $toolsDir)) {
        New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
    }
    $jarUrl = "https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar"
    Invoke-WebRequest -Uri $jarUrl -OutFile $ResolvedJar
}

$ResolvedJar = (Resolve-Path $ResolvedJar).Path
Write-Host "    PlantUML jar: $ResolvedJar" -ForegroundColor Green


# ── 3. Check Graphviz / Layout Engine ─────────────────────────────────────────

Write-Host "==> Checking Graphviz / Smetana layout engine..." -ForegroundColor Cyan
$pumlVer = Invoke-CliProcess -FilePath $ResolvedJava -ArgumentList @("-jar", "`"$ResolvedJar`"", "-version")
$graphvizStatus = ($pumlVer.Combined -split "`r?`n" | Where-Object { $_ -match "GraphViz" -or $_ -match "Installation" }) -join "; "
Write-Host "    $graphvizStatus" -ForegroundColor Green


# ── 4. Prepare Directories ────────────────────────────────────────────────────

$ResolvedSource = Join-Path $PSScriptRoot "..\$SourceDir"
if (Test-Path $ResolvedSource) {
    $ResolvedSource = (Resolve-Path $ResolvedSource).Path
} else {
    $ResolvedSource = (Resolve-Path $SourceDir).Path
}

$ResolvedOutput = Join-Path $PSScriptRoot "..\$OutputDir"
if (-not (Test-Path $ResolvedOutput)) {
    New-Item -ItemType Directory -Force -Path $ResolvedOutput | Out-Null
}
$ResolvedOutput = (Resolve-Path $ResolvedOutput).Path

Write-Host "==> Source directory: $ResolvedSource" -ForegroundColor Cyan
Write-Host "==> Output directory: $ResolvedOutput" -ForegroundColor Cyan


# ── 5. Determine Render Formats ───────────────────────────────────────────────

$targetFormats = @()
if ($Format -eq "all" -or $IncludePng) {
    $targetFormats += "svg"
    $targetFormats += "png"
} elseif ($Format -eq "png") {
    $targetFormats += "png"
} else {
    $targetFormats += "svg"
}


# ── 6. Render Diagrams ────────────────────────────────────────────────────────

$pumlFiles = Get-ChildItem -Path $ResolvedSource -Filter "*.puml" | Sort-Object Name

if ($pumlFiles.Count -eq 0) {
    Write-Warning "No .puml files found in $ResolvedSource."
    exit 0
}

Write-Host "==> Found $($pumlFiles.Count) diagram(s) to render:" -ForegroundColor Cyan
foreach ($f in $pumlFiles) {
    Write-Host "    - $($f.Name)"
}

$successCount = 0
$totalRenderJobs = $pumlFiles.Count * $targetFormats.Count

foreach ($fmt in $targetFormats) {
    Write-Host "`n==> Rendering format: $($fmt.ToUpper())..." -ForegroundColor Yellow
    foreach ($file in $pumlFiles) {
        Write-Host "    Rendering $($file.Name) -> $fmt..." -NoNewline

        $argsList = @(
            "-DPLANTUML_LIMIT_SIZE=8192",
            "-jar", "`"$ResolvedJar`"",
            "-t$fmt",
            "-o", "`"$ResolvedOutput`"",
            "`"$($file.FullName)`""
        )

        $res = Invoke-CliProcess -FilePath $ResolvedJava -ArgumentList $argsList

        if ($res.ExitCode -eq 0) {
            Write-Host " [OK]" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host " [FAILED (Exit Code $($res.ExitCode))]" -ForegroundColor Red
            if ($res.Combined) {
                Write-Host "      $($res.Combined)" -ForegroundColor DarkRed
            }
        }
    }
}


# ── 7. Verification & Summary ─────────────────────────────────────────────────

Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host " VIDEX PlantUML Rendering Summary" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host " Total jobs completed: $successCount of $totalRenderJobs"

$renderedSvg = Get-ChildItem -Path $ResolvedOutput -Filter "*.svg"
$renderedPng = Get-ChildItem -Path $ResolvedOutput -Filter "*.png"

Write-Host " Rendered SVGs ($($renderedSvg.Count)):"
foreach ($svg in $renderedSvg) {
    $sizeKb = [math]::Round($svg.Length / 1KB, 1)
    Write-Host "   + $($svg.Name) ($sizeKb KB)" -ForegroundColor Green
}

if ($renderedPng.Count -gt 0) {
    Write-Host " Rendered PNGs ($($renderedPng.Count)):"
    foreach ($png in $renderedPng) {
        $sizeKb = [math]::Round($png.Length / 1KB, 1)
        Write-Host "   + $($png.Name) ($sizeKb KB)" -ForegroundColor Green
    }
}

Write-Host ("=" * 60) -ForegroundColor Cyan

if ($renderedSvg.Count -lt $pumlFiles.Count) {
    Write-Host "[ERROR] Expected $($pumlFiles.Count) SVG files, but found $($renderedSvg.Count)." -ForegroundColor Red
    exit 1
}

Write-Host "`nAll diagrams rendered successfully into: $ResolvedOutput`n" -ForegroundColor Green
exit 0
