$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$jar = Join-Path $serviceRoot 'target\document-api-0.1.0.jar'
$mysqlConfig = Join-Path $serviceRoot '.local\mysql.json'
$minioConfig = Join-Path $serviceRoot '.local\minio.json'

foreach ($required in @($jar, $mysqlConfig, $minioConfig)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing: $required" }
}

$jdk = [Environment]::GetEnvironmentVariable('JAVA_HOME', 'User')
if (-not $jdk) { $jdk = $env:JAVA_HOME }
if (-not $jdk) { $jdk = 'D:\yingwen\jdk-17.0.21+9' }
$java = Join-Path $jdk 'bin\java.exe'
if (-not (Test-Path -LiteralPath $java)) { throw "JDK 17 is missing: $java" }

$mysql = Get-Content -LiteralPath $mysqlConfig -Raw | ConvertFrom-Json
$minio = Get-Content -LiteralPath $minioConfig -Raw | ConvertFrom-Json
$ingestConfig = Get-Content -LiteralPath (& (Join-Path $PSScriptRoot 'ensure-ingest-config.ps1')) -Raw | ConvertFrom-Json
$env:MYSQL_URL = "jdbc:mysql://127.0.0.1:$($mysql.port)/modular_rag?connectionTimeZone=UTC"
$env:MYSQL_USER = $mysql.appUser
$env:MYSQL_PASSWORD = $mysql.appPassword
$env:MINIO_ENDPOINT = 'http://127.0.0.1:9000'
$env:MINIO_ACCESS_KEY = $minio.accessKey
$env:MINIO_SECRET_KEY = $minio.secretKey
$env:MINIO_BUCKET = 'rag-documents'
$env:SERVER_ADDRESS = '127.0.0.1'
$env:KAFKA_BOOTSTRAP_SERVERS = '127.0.0.1:9092'
$env:PYTHON_INGEST_URL = 'http://127.0.0.1:8090/internal/ingest'
$env:PYTHON_INGEST_TOKEN = $ingestConfig.token

$process = Start-Process -FilePath $java -ArgumentList @('-jar', $jar) `
    -WorkingDirectory $serviceRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $serviceRoot '.local\app.stdout.log') `
    -RedirectStandardError (Join-Path $serviceRoot '.local\app.stderr.log')
Write-Output "Java API started with PID $($process.Id) on 127.0.0.1:8080"
