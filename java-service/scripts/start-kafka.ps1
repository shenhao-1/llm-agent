$ErrorActionPreference = 'Stop'
$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$kafkaRoot = Join-Path $serviceRoot '.tools\kafka_2.13-3.9.2'
$libs = Join-Path $kafkaRoot 'libs\*'
if (-not (Test-Path -LiteralPath (Join-Path $kafkaRoot 'libs\kafka-storage-3.9.2.jar'))) {
    throw "Kafka 3.9.2 is missing under $kafkaRoot; see java-service/README.md"
}
$jdk = [Environment]::GetEnvironmentVariable('JAVA_HOME', 'User')
if (-not $jdk) { $jdk = 'D:\yingwen\jdk-17.0.21+9' }
$env:JAVA_HOME = $jdk
$env:Path = "$(Join-Path $jdk 'bin');$env:Path"
$java = Join-Path $jdk 'bin\java.exe'
$config = Join-Path $serviceRoot '.local\kafka.properties'
$dataDir = (Join-Path $serviceRoot '.local\kafka-data').Replace('\', '/')
if (-not (Test-Path -LiteralPath $config)) {
    @"
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@127.0.0.1:9093
listeners=PLAINTEXT://127.0.0.1:9092,CONTROLLER://127.0.0.1:9093
advertised.listeners=PLAINTEXT://127.0.0.1:9092
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
controller.listener.names=CONTROLLER
inter.broker.listener.name=PLAINTEXT
log.dirs=$dataDir
num.partitions=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
group.initial.rebalance.delay.ms=0
auto.create.topics.enable=false
"@ | Set-Content -LiteralPath $config -Encoding ascii
}
if (-not (Test-Path -LiteralPath (Join-Path $serviceRoot '.local\kafka-data\meta.properties'))) {
    $clusterId = (& $java -cp $libs kafka.tools.StorageTool random-uuid | Select-Object -Last 1).Trim()
    & $java -cp $libs kafka.tools.StorageTool format -t $clusterId -c $config
    if ($LASTEXITCODE -ne 0) { throw 'Kafka storage format failed' }
}
$log4j = (Join-Path $kafkaRoot 'config\log4j.properties').Replace('\', '/')
$process = Start-Process -FilePath $java `
    -ArgumentList @('-Xms512m', '-Xmx512m', "-Dlog4j.configuration=file:$log4j", '-cp', $libs, 'kafka.Kafka', $config) `
    -WorkingDirectory $kafkaRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $serviceRoot '.local\kafka.stdout.log') `
    -RedirectStandardError (Join-Path $serviceRoot '.local\kafka.stderr.log')
Write-Output "Kafka launcher started with PID $($process.Id) on 127.0.0.1:9092"
