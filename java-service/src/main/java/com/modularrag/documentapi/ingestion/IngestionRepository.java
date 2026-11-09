package com.modularrag.documentapi.ingestion;

import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
public class IngestionRepository {
    private final JdbcTemplate jdbc;

    public IngestionRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public record OutboxEvent(long id, String taskId, int attemptNo) {
    }

    public record WorkItem(String taskId, String documentId, String collection, String objectKey,
                           int attemptCount) {
    }

    public List<OutboxEvent> dueEvents(Instant now) {
        return jdbc.query("""
                SELECT id, task_id, attempt_no FROM ingestion_outbox
                WHERE published_at IS NULL AND available_at <= ?
                ORDER BY id LIMIT 20
                """, (rs, row) -> new OutboxEvent(rs.getLong(1), rs.getString(2), rs.getInt(3)),
                Timestamp.from(now));
    }

    public void markPublished(long eventId, Instant now) {
        jdbc.update("UPDATE ingestion_outbox SET published_at = ? WHERE id = ? AND published_at IS NULL",
                Timestamp.from(now), eventId);
    }

    public WorkItem claim(String taskId, int attemptNo, Instant now, Instant leaseUntil) {
        int claimed = jdbc.update("""
                UPDATE ingestion_tasks SET status = 'PROCESSING', attempt_count = attempt_count + 1,
                    lease_until = ?, error_message = NULL, updated_at = ?
                WHERE id = ? AND attempt_count = ? AND status IN ('PENDING', 'RETRY')
                """, Timestamp.from(leaseUntil), Timestamp.from(now), taskId, attemptNo);
        if (claimed == 0) {
            return null;
        }
        jdbc.update("UPDATE documents SET status = 'PROCESSING', updated_at = ? "
                + "WHERE id = (SELECT document_id FROM ingestion_tasks WHERE id = ?)",
                Timestamp.from(now), taskId);
        return jdbc.queryForObject("""
                SELECT t.id, t.document_id, d.collection_name, d.object_key, t.attempt_count
                FROM ingestion_tasks t JOIN documents d ON d.id = t.document_id WHERE t.id = ?
                """, (rs, row) -> new WorkItem(rs.getString(1), rs.getString(2),
                        rs.getString(3), rs.getString(4), rs.getInt(5)), taskId);
    }

    public void complete(WorkItem item, Instant now) {
        int changed = jdbc.update("""
                UPDATE ingestion_tasks SET status = 'COMPLETED', lease_until = NULL, updated_at = ?
                WHERE id = ? AND status = 'PROCESSING' AND attempt_count = ?
                """, Timestamp.from(now), item.taskId(), item.attemptCount());
        if (changed == 1) {
            jdbc.update("UPDATE documents SET status = 'READY', updated_at = ? WHERE id = ?",
                    Timestamp.from(now), item.documentId());
        }
    }

    public void fail(WorkItem item, String reason, Instant now, int maxAttempts) {
        boolean terminal = item.attemptCount() >= maxAttempts;
        int changed = jdbc.update("""
                UPDATE ingestion_tasks SET status = ?, error_message = ?, lease_until = NULL,
                    updated_at = ? WHERE id = ? AND status = 'PROCESSING' AND attempt_count = ?
                """, terminal ? "FAILED" : "RETRY", reason, Timestamp.from(now),
                item.taskId(), item.attemptCount());
        if (changed != 1) {
            return;
        }
        if (terminal) {
            jdbc.update("UPDATE documents SET status = 'FAILED', updated_at = ? WHERE id = ?",
                    Timestamp.from(now), item.documentId());
        } else {
            scheduleRetry(item.taskId(), item.attemptCount(), now);
        }
    }

    public List<WorkItem> expired(Instant now) {
        return jdbc.query("""
                SELECT t.id, t.document_id, d.collection_name, d.object_key, t.attempt_count
                FROM ingestion_tasks t JOIN documents d ON d.id = t.document_id
                WHERE t.status = 'PROCESSING' AND t.lease_until < ? LIMIT 20
                """, (rs, row) -> new WorkItem(rs.getString(1), rs.getString(2),
                        rs.getString(3), rs.getString(4), rs.getInt(5)), Timestamp.from(now));
    }

    public void recover(WorkItem item, Instant now, int maxAttempts) {
        boolean terminal = item.attemptCount() >= maxAttempts;
        int changed = jdbc.update("""
                UPDATE ingestion_tasks SET status = ?, error_message = 'Worker lease expired',
                    lease_until = NULL, updated_at = ?
                WHERE id = ? AND status = 'PROCESSING' AND attempt_count = ? AND lease_until < ?
                """, terminal ? "FAILED" : "RETRY", Timestamp.from(now), item.taskId(),
                item.attemptCount(), Timestamp.from(now));
        if (changed != 1) {
            return;
        }
        if (terminal) {
            jdbc.update("UPDATE documents SET status = 'FAILED', updated_at = ? WHERE id = ?",
                    Timestamp.from(now), item.documentId());
        } else {
            scheduleRetry(item.taskId(), item.attemptCount(), now);
        }
    }

    private void scheduleRetry(String taskId, int attemptNo, Instant now) {
        long delaySeconds = Math.min(60L, 5L << Math.min(4, attemptNo - 1));
        jdbc.update("""
                INSERT INTO ingestion_outbox (task_id, attempt_no, available_at, created_at)
                VALUES (?, ?, ?, ?)
                """, taskId, attemptNo, Timestamp.from(now.plusSeconds(delaySeconds)),
                Timestamp.from(now));
    }
}
