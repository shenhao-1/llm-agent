package com.modularrag.documentapi.document;

import java.time.Instant;

public record TaskRecord(
        String id,
        String documentId,
        String status,
        int attemptCount,
        String errorMessage,
        Instant createdAt,
        Instant updatedAt) {
}
