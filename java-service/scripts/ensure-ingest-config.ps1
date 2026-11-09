$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$configPath = Join-Path $serviceRoot '.local\ingest.json'
if (-not (Test-Path -LiteralPath $configPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $configPath) | Out-Null
    $bytes = New-Object byte[] 32
    $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $generator.GetBytes($bytes) } finally { $generator.Dispose() }
    $config = @{ token = [BitConverter]::ToString($bytes).Replace('-', '').ToLowerInvariant() }
    $config | ConvertTo-Json | Set-Content -LiteralPath $configPath -Encoding utf8
}
$configPath
