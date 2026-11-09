package com.modularrag.documentapi.document;

import java.time.Instant;

public record DocumentRecord(
        String id,
        String collection,
        String fileName,
        String objectKey,
        String contentType,
        long sizeBytes,
        String status,
        Instant createdAt,
        Instant updatedAt) {
}
