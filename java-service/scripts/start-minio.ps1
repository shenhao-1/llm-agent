$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$binary = Join-Path $serviceRoot '.tools\minio.exe'
$configPath = Join-Path $serviceRoot '.local\minio.json'
$dataPath = Join-Path $serviceRoot '.local\minio-data'

if (-not (Test-Path -LiteralPath $binary)) {
    throw "MinIO binary missing: $binary"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "MinIO local credentials missing: $configPath"
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$env:MINIO_ROOT_USER = $config.accessKey
$env:MINIO_ROOT_PASSWORD = $config.secretKey
New-Item -ItemType Directory -Force -Path $dataPath | Out-Null

$process = Start-Process -FilePath $binary `
    -ArgumentList @('server', $dataPath, '--address', '127.0.0.1:9000', '--console-address', '127.0.0.1:9001') `
    -WorkingDirectory $serviceRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $serviceRoot '.local\minio.stdout.log') `
    -RedirectStandardError (Join-Path $serviceRoot '.local\minio.stderr.log')
Write-Output "MinIO started with PID $($process.Id) on 127.0.0.1:9000"
