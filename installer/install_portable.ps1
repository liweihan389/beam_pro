$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw '此安装程序仅用于 Windows。' }
$Source = $PSScriptRoot
$Destination = Join-Path $env:LOCALAPPDATA 'Programs\beamic-pro'
if ([IO.Path]::GetFullPath($Source).TrimEnd('\') -ne [IO.Path]::GetFullPath($Destination).TrimEnd('\')) {
    New-Item -ItemType Directory -Force $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source | Copy-Item -Destination $Destination -Recurse -Force
}
$StartMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'beamic pro'
New-Item -ItemType Directory -Force $StartMenu | Out-Null
$Shell = New-Object -ComObject WScript.Shell
foreach ($Name in @('beamic pro', 'beamic pro 模拟光路')) {
    $Shortcut = $Shell.CreateShortcut((Join-Path $StartMenu "$Name.lnk"))
    $Shortcut.TargetPath = Join-Path $Destination 'runtime\pythonw.exe'
    $Shortcut.Arguments = '"' + (Join-Path $Destination 'beamic-pro.pyw') + '"'
    if ($Name -like '*模拟*') { $Shortcut.Arguments += ' --demo' }
    $Shortcut.WorkingDirectory = $Destination
    $Shortcut.Save()
}
Write-Output "安装完成：$Destination"
Write-Output '可从开始菜单打开 beamic pro。真实相机接入需要原版 BeamMic 和 .NET Framework 4.8。'

