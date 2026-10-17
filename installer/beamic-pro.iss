#ifndef MyAppVersion
  #define MyAppVersion "0.5.1"
#endif

[Setup]
AppId={{AD90CE46-936B-4319-9BA6-538A2370C790}
AppName=beamic pro
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\Programs\beamic-pro
DefaultGroupName=beamic pro
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=beamic-pro-{#MyAppVersion}-setup
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\beamic-pro.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\dist\beamic-pro\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\beamic pro"; Filename: "{app}\beamic-pro.exe"
Name: "{group}\模拟光路"; Filename: "{app}\beamic-pro.exe"; Parameters: "--demo"

[Run]
Filename: "{app}\beamic-pro.exe"; Description: "打开 beamic pro"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var Release: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full', 'Release', Release);
  if Result then Result := Release >= 528040;
  if not Result then MsgBox('需要先安装 Microsoft .NET Framework 4.8 或更高版本，再运行安装程序。', mbError, MB_OK);
end;

