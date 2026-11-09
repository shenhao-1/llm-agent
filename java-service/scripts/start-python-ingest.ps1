$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoRoot = (Resolve-Path (Join-Path $serviceRoot '..')).Path
$minio = Get-Content -LiteralPath (Join-Path $serviceRoot '.local\minio.json') -Raw | ConvertFrom-Json
$ingestConfig = Get-Content -LiteralPath (& (Join-Path $PSScriptRoot 'ensure-ingest-config.ps1')) -Raw | ConvertFrom-Json
$env:MINIO_ENDPOINT = 'http://127.0.0.1:9000'
$env:MINIO_ACCESS_KEY = $minio.accessKey
$env:MINIO_SECRET_KEY = $minio.secretKey
$env:MINIO_BUCKET = 'rag-documents'
$env:PYTHON_INGEST_TOKEN = $ingestConfig.token
$env:INGEST_OFFLINE = 'true'
$env:INGEST_DATA_DIR = Join-Path $serviceRoot '.local\python-data'
$env:PYTHONPATH = "$(Join-Path $serviceRoot '.tools\python');$repoRoot;$(Join-Path $repoRoot 'src')"
$env:PYTHON_INGEST_HOST = '127.0.0.1'
$env:PYTHON_INGEST_PORT = '8090'
$python = (Get-Command python -ErrorAction Stop).Source
$process = Start-Process -FilePath $python -ArgumentList @('-m', 'scripts.internal_ingest_api') `
    -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $serviceRoot '.local\python.stdout.log') `
    -RedirectStandardError (Join-Path $serviceRoot '.local\python.stderr.log')
Write-Output "Python ingest API started with PID $($process.Id) on 127.0.0.1:8090"
