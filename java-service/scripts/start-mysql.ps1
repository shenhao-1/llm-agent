$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$binary = 'C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqld.exe'
$basePath = 'C:\Program Files\MySQL\MySQL Server 8.0'
$dataPath = Join-Path $serviceRoot '.local\mysql-data'

if (-not (Test-Path -LiteralPath $binary)) {
    throw "MySQL server binary missing: $binary"
}
if (-not (Test-Path -LiteralPath (Join-Path $dataPath 'ibdata1'))) {
    throw "MySQL data directory is not initialized: $dataPath"
}

$process = Start-Process -FilePath $binary `
    -ArgumentList @('--no-defaults', "--basedir=`"$basePath`"", "--datadir=$dataPath", '--port=3307', '--bind-address=127.0.0.1', '--mysqlx=OFF', "--log-error=$(Join-Path $serviceRoot '.local\mysql.err')") `
    -WorkingDirectory $serviceRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $serviceRoot '.local\mysql.stdout.log') `
    -RedirectStandardError (Join-Path $serviceRoot '.local\mysql.stderr.log')
Write-Output "MySQL started with PID $($process.Id) on 127.0.0.1:3307"
