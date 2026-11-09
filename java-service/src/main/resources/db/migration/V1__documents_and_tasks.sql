CREATE TABLE documents (
    id CHAR(36) NOT NULL PRIMARY KEY,
    collection_name VARCHAR(100) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    object_key VARCHAR(512) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    size_bytes BIGINT NOT NULL,
    status VARCHAR(32) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    UNIQUE KEY uk_documents_object_key (object_key),
    KEY idx_documents_created_at (created_at)
);

CREATE TABLE ingestion_tasks (
    id CHAR(36) NOT NULL PRIMARY KEY,
    document_id CHAR(36) NOT NULL,
    status VARCHAR(32) NOT NULL,
    attempt_count INT NOT NULL DEFAULT 0,
    error_message VARCHAR(1000) NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    CONSTRAINT fk_ingestion_tasks_document FOREIGN KEY (document_id) REFERENCES documents(id),
    KEY idx_tasks_document_id (document_id),
    KEY idx_tasks_status (status)
);
