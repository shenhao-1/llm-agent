# Java document API and asynchronous ingestion

This Spring Boot service adds PDF upload, document listing, and task status to the
Python RAG project. The PDF bytes go to MinIO. MySQL stores document metadata and
an ingestion task with status `PENDING`. A transactional MySQL outbox publishes
the task ID and attempt number to Kafka. A Java consumer calls the private Python
ingestion API, which fetches the PDF from MinIO and runs the existing PDF → chunk
→ embedding → BM25/vector pipeline. Elasticsearch and Redis are not integrated.

The Python pipeline uses the configured embedding provider by default. The local
Windows and Compose examples set `INGEST_OFFLINE=true`, which runs the project's
deterministic demo embedding and JSON vector store. This verifies the full async
flow without external model credentials; it is not a production semantic index.

## Requirements

- Docker Compose, or JDK 17+, Maven 3.6.3+, MySQL, MinIO, Kafka 3.9.2, and
  Python 3.11+ for a manual run.
- For local development with Docker (builds the Java application and starts all
  services):

  ```powershell
  cd java-service
  docker compose up --build -d
  ```

The Compose setup builds MinIO from its pinned official GitHub release asset
because the old community binary download endpoint is no longer available.
The Compose defaults are **development credentials**. Change them before using
this service outside a local machine. The Spring configuration reads `MYSQL_URL`,
`MYSQL_USER`, `MYSQL_PASSWORD`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
`MINIO_SECRET_KEY`, and `MINIO_BUCKET`. Flyway creates the tables on startup.
MinIO creates the bucket on the first successful upload.

## This Windows workspace without Docker

The local environment is installed as follows:

- JDK 17: `D:\yingwen\jdk-17.0.21+9`; the current user's `JAVA_HOME` points there.
- MySQL development instance: `127.0.0.1:3307`, with data in `.local/mysql-data`.
  This is separate from the pre-existing MySQL service on port 3306.
- MinIO development instance: `127.0.0.1:9000`, console on port 9001, data in
  `.local/minio-data`. Its official Windows binary is in `.tools/minio.exe`.
- The application: `127.0.0.1:8080`. MySQL and MinIO credentials are generated
  randomly and stored under `.local`; that directory is ignored by Git.
- Kafka: `127.0.0.1:9092`, installed under `.tools/kafka_2.13-3.9.2` and using
  `.local/kafka-data`.
- Python ingestion API: `127.0.0.1:8090`. Java and Python share an internal
  token generated in ignored `.local/ingest.json`.

Install the Python MinIO client into the ignored local tools directory once:

```powershell
python -m pip install --target java-service/.tools/python "minio>=7,<8"
```

Download the official Kafka 3.9.2 binary and verify its `.sha512` file before
extracting it to `java-service/.tools/kafka_2.13-3.9.2`. The local start script
initializes a single-node KRaft broker on its first run. It binds only to
loopback for this development environment.

After a reboot, start each stopped component from the repository root:

```powershell
.\java-service\scripts\start-mysql.ps1
.\java-service\scripts\start-minio.ps1
.\java-service\scripts\start-kafka.ps1
.\java-service\scripts\start-python-ingest.ps1
.\java-service\scripts\start-app.ps1
```

To rebuild or test the Java application, use
`.\java-service\scripts\build-local.ps1 package` or
`.\java-service\scripts\build-local.ps1 test`. The script selects JDK 17 and
keeps Maven dependencies inside the ignored `java-service/.m2` directory.
If the Java source changes, rebuild before restarting the application.

The native MinIO community release is archived; use this setup only for local
development. The local services bind to loopback addresses.

The API listens on `http://localhost:8080` by default.

```powershell
curl.exe -F "file=@sample.pdf;type=application/pdf" -F "collection=docs" http://localhost:8080/api/documents
curl.exe "http://localhost:8080/api/documents?limit=20&offset=0"
curl.exe http://localhost:8080/api/documents/DOCUMENT_ID
curl.exe http://localhost:8080/api/tasks/TASK_ID
curl.exe http://localhost:8080/api/documents/DOCUMENT_ID/tasks
```

`POST /api/documents` returns HTTP 201 with `documentId`, `taskId`,
`documentStatus=UPLOADED`, and `taskStatus=PENDING`. Kafka processing then moves
the task through `PROCESSING` to `COMPLETED`, or `RETRY` and finally `FAILED`.
PDF files are limited to
50 MB. The first five bytes must be `%PDF-`. The public list and detail APIs do
not expose MinIO object keys. Database insertions are transactional; if they
fail after upload, the service attempts to remove the MinIO object.

The outbox keeps upload successful if Kafka is temporarily down and retries
publication when it returns. The consumer claims a task with a conditional
MySQL update keyed by `(task ID, attempt number)`, so duplicate Kafka deliveries
cannot start the same attempt twice. The Python pipeline uses a stable local
source path and its existing SHA-256 ingestion history to skip repeated successful
work. Failures are retried after increasing delays, up to three attempts; an
expired worker lease is recovered. MySQL and Kafka are therefore **at least once**,
with task-level idempotency rather than a cross-system exactly-once transaction.

This phase has no public API authentication or tenant isolation. Add those before
exposing the API to untrusted users. The Python endpoint requires the shared
internal token and should remain on a private network.
