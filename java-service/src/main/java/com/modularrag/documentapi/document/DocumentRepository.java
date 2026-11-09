package com.modularrag.documentapi.document;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class DocumentRepository {
    private final JdbcTemplate jdbc;

    public DocumentRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public void insert(DocumentRecord document, TaskRecord task) {
        jdbc.update("""
                INSERT INTO documents
                    (id, collection_name, original_name, object_key, content_type,
                     size_bytes, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                document.id(), document.collection(), document.fileName(), document.objectKey(),
                document.contentType(), document.sizeBytes(), document.status(),
                Timestamp.from(document.createdAt()), Timestamp.from(document.updatedAt()));
        jdbc.update("""
                INSERT INTO ingestion_tasks
                    (id, document_id, status, attempt_count, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                task.id(), task.documentId(), task.status(), task.attemptCount(),
                task.errorMessage(), Timestamp.from(task.createdAt()), Timestamp.from(task.updatedAt()));
        jdbc.update("""
                INSERT INTO ingestion_outbox (task_id, attempt_no, available_at, created_at)
                VALUES (?, 0, ?, ?)
                """, task.id(), Timestamp.from(task.createdAt()), Timestamp.from(task.createdAt()));
    }

    public List<DocumentRecord> list(int limit, int offset) {
        return jdbc.query("""
                SELECT id, collection_name, original_name, object_key, content_type,
                       size_bytes, status, created_at, updated_at
                FROM documents ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?
                """, this::mapDocument, limit, offset);
    }

    public long count() {
        Long count = jdbc.queryForObject("SELECT COUNT(*) FROM documents", Long.class);
        return count == null ? 0 : count;
    }

    public Optional<DocumentRecord> findDocument(String id) {
        return jdbc.query("""
                SELECT id, collection_name, original_name, object_key, content_type,
                       size_bytes, status, created_at, updated_at
                FROM documents WHERE id = ?
                """, this::mapDocument, id).stream().findFirst();
    }

    public Optional<TaskRecord> findTask(String id) {
        return jdbc.query("""
                SELECT id, document_id, status, attempt_count, error_message,
                       created_at, updated_at
                FROM ingestion_tasks WHERE id = ?
                """, this::mapTask, id).stream().findFirst();
    }

    public List<TaskRecord> findTasksByDocument(String documentId) {
        return jdbc.query("""
                SELECT id, document_id, status, attempt_count, error_message,
                       created_at, updated_at
                FROM ingestion_tasks WHERE document_id = ? ORDER BY created_at DESC, id DESC
                """, this::mapTask, documentId);
    }

    private DocumentRecord mapDocument(ResultSet rs, int row) throws SQLException {
        return new DocumentRecord(
                rs.getString("id"), rs.getString("collection_name"),
                rs.getString("original_name"), rs.getString("object_key"),
                rs.getString("content_type"), rs.getLong("size_bytes"),
                rs.getString("status"), toInstant(rs, "created_at"),
                toInstant(rs, "updated_at"));
    }

    private TaskRecord mapTask(ResultSet rs, int row) throws SQLException {
        return new TaskRecord(
                rs.getString("id"), rs.getString("document_id"), rs.getString("status"),
                rs.getInt("attempt_count"), rs.getString("error_message"),
                toInstant(rs, "created_at"), toInstant(rs, "updated_at"));
    }

    private Instant toInstant(ResultSet rs, String column) throws SQLException {
        return rs.getTimestamp(column).toInstant();
    }
}
