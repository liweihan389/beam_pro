param([switch]$SkipInstaller, [switch]$IncludeDocs)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Build command failed: $Executable (exit $LASTEXITCODE)" }
}
if (-not $IsWindows -and $env:OS -ne 'Windows_NT') { throw 'Build this application on Windows x64.' }
Invoke-Checked 'python' @('-c', 'import sys; assert sys.version_info[:2] == (3,12), "Use CPython 3.12 x64"; assert sys.maxsize > 2**32')
Invoke-Checked 'python' @('-m', 'venv', '.build-venv')
$Python = Join-Path $Root '.build-venv\Scripts\python.exe'
Invoke-Checked $Python @('-m', 'pip', 'install', '-r', 'requirements-core.lock', '-e', 'sdk', '-e', '.[dev]')
Invoke-Checked $Python @('-m', 'pytest', '-q')
$Version = (& $Python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])').Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot read application version.' }
Invoke-Checked 'dotnet' @('build', 'bridge\BeamMicBridge.csproj', '-c', 'Release')
Invoke-Checked 'dotnet' @('run', '--project', 'bridge-tests\BeamMicBridge.Tests.csproj', '-c', 'Release', '-f', 'net48')
Invoke-Checked $Python @('-m', 'build', 'sdk', '--wheel', '--outdir', 'build\wheels')
Invoke-Checked $Python @('scripts\package_plugins.py')
$Separator = ';'
Invoke-Checked $Python @('-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed', '--name', 'beamic-pro',
    '--add-data', "plugins${Separator}assets\plugins", '--add-data', "build\wheels${Separator}assets\wheels",
    '--add-data', "bridge\device-profile.example.json${Separator}assets\bridge", 'scripts\launch.py')
Invoke-Checked $Python @('-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile', '--console', '--name', 'beamic-preflight', 'scripts\preflight_launch.py')
$Application = Join-Path $Root 'dist\beamic-pro'
New-Item -ItemType Directory -Force (Join-Path $Application 'bridge') | Out-Null
Copy-Item 'bridge\bin\Release\net48\*' (Join-Path $Application 'bridge') -Force
Copy-Item 'dist\beamic-preflight.exe' $Application -Force
Copy-Item 'README.md' $Application -Force
if ($IncludeDocs) { Copy-Item 'docs' $Application -Recurse -Force }
# Full private CPython supplies venv and ensurepip for the shared plugin environment.
$Base = (& $Python -c 'import sys; print(sys.base_prefix)').Trim()
$Runtime = Join-Path $Application 'runtime'
New-Item -ItemType Directory -Force $Runtime | Out-Null
foreach ($name in @('python.exe','pythonw.exe','python3.dll','python312.dll','vcruntime140.dll','vcruntime140_1.dll','LICENSE.txt')) {
    if (Test-Path (Join-Path $Base $name)) { Copy-Item (Join-Path $Base $name) $Runtime -Force }
}
Copy-Item (Join-Path $Base 'DLLs') $Runtime -Recurse -Force
$Lib = Join-Path $Runtime 'Lib'
New-Item -ItemType Directory -Force $Lib | Out-Null
Get-ChildItem (Join-Path $Base 'Lib') | Where-Object { $_.Name -notin @('site-packages','__pycache__') } |
    Copy-Item -Destination $Lib -Recurse -Force
Invoke-Checked (Join-Path $Runtime 'python.exe') @('-c', 'import venv, ensurepip; print("Private plugin runtime ready")')
if (-not $SkipInstaller) {
    $Compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if (-not $Compiler) {
        $Candidate = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
        if (Test-Path $Candidate) { $Compiler = $Candidate } else { throw 'Install Inno Setup 6, or use -SkipInstaller for a portable build.' }
    }
    Invoke-Checked "$Compiler" @("/DMyAppVersion=$Version", 'installer\beamic-pro.iss')
}
Compress-Archive -Path "$Application\*" -DestinationPath 'dist\beamic-pro-windows-portable.zip' -Force
Write-Output 'Build finished. Real BeamMic/camera and alignment acceptance must still be run on the laboratory computer.'

