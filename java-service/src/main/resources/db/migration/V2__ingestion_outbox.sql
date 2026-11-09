ALTER TABLE ingestion_tasks ADD COLUMN lease_until DATETIME(6) NULL;

CREATE TABLE ingestion_outbox (
    id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    task_id CHAR(36) NOT NULL,
    attempt_no INT NOT NULL,
    available_at DATETIME(6) NOT NULL,
    published_at DATETIME(6) NULL,
    created_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_outbox_task FOREIGN KEY (task_id) REFERENCES ingestion_tasks(id),
    UNIQUE KEY uk_outbox_task_attempt (task_id, attempt_no),
    KEY idx_outbox_due (published_at, available_at)
);

INSERT INTO ingestion_outbox (task_id, attempt_no, available_at, created_at)
SELECT id, 0, UTC_TIMESTAMP(6), UTC_TIMESTAMP(6)
FROM ingestion_tasks WHERE status = 'PENDING';
