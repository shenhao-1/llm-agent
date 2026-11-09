param([string]$Goal = 'test')

$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$jdk = [Environment]::GetEnvironmentVariable('JAVA_HOME', 'User')
if (-not $jdk) { $jdk = $env:JAVA_HOME }
if (-not $jdk) { $jdk = 'D:\yingwen\jdk-17.0.21+9' }
$java = Join-Path $jdk 'bin\java.exe'
if (-not (Test-Path -LiteralPath $java)) { throw "JDK 17 is missing: $java" }

$env:JAVA_HOME = $jdk
$env:Path = "$(Join-Path $jdk 'bin');$env:Path"
$repository = (Join-Path $serviceRoot '.m2').Replace('\', '/')
$settingsPath = Join-Path $serviceRoot '.tools\settings.xml'
New-Item -ItemType Directory -Force -Path (Split-Path $settingsPath) | Out-Null
@"
<?xml version="1.0" encoding="UTF-8"?>
<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0">
  <localRepository>$repository</localRepository>
</settings>
"@ | Set-Content -LiteralPath $settingsPath -Encoding utf8

Push-Location $serviceRoot
try {
    & mvn -s $settingsPath $Goal
    if ($LASTEXITCODE -ne 0) { throw "Maven $Goal failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
