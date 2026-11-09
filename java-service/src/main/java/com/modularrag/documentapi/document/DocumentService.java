package com.modularrag.documentapi.document;

import com.modularrag.documentapi.storage.ObjectStorage;
import java.io.IOException;
import java.io.InputStream;
import java.time.Clock;
import java.time.Instant;
import java.util.List;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.web.multipart.MultipartFile;

@Service
public class DocumentService {
    private final DocumentRepository repository;
    private final ObjectStorage storage;
    private final TransactionTemplate transactions;
    private final Clock clock;

    @Autowired
    public DocumentService(DocumentRepository repository, ObjectStorage storage,
                           TransactionTemplate transactions) {
        this(repository, storage, transactions, Clock.systemUTC());
    }

    DocumentService(DocumentRepository repository, ObjectStorage storage,
                    TransactionTemplate transactions, Clock clock) {
        this.repository = repository;
        this.storage = storage;
        this.transactions = transactions;
        this.clock = clock;
    }

    public UploadResult upload(MultipartFile file, String collection) {
        validatePdf(file);
        String normalizedCollection = normalizeCollection(collection);
        String fileName = normalizeFileName(file.getOriginalFilename());
        String documentId = UUID.randomUUID().toString();
        String taskId = UUID.randomUUID().toString();
        String objectKey = "documents/" + documentId + "/original.pdf";
        Instant now = clock.instant();

        try (InputStream input = file.getInputStream()) {
            storage.put(objectKey, input, file.getSize());
        } catch (IOException e) {
            throw new IllegalArgumentException("Could not read uploaded PDF", e);
        } catch (RuntimeException storageFailure) {
            removeAfterFailure(objectKey, storageFailure);
            throw storageFailure;
        }

        DocumentRecord document = new DocumentRecord(documentId, normalizedCollection,
                fileName, objectKey, "application/pdf", file.getSize(), "UPLOADED", now, now);
        TaskRecord task = new TaskRecord(taskId, documentId, "PENDING", 0, null, now, now);
        try {
            transactions.executeWithoutResult(status -> repository.insert(document, task));
        } catch (RuntimeException dbFailure) {
            removeAfterFailure(objectKey, dbFailure);
            throw dbFailure;
        }
        return new UploadResult(documentId, taskId, document.status(), task.status());
    }

    public List<DocumentRecord> list(int limit, int offset) {
        if (limit < 1 || limit > 100 || offset < 0) {
            throw new IllegalArgumentException("limit must be 1..100 and offset must be >= 0");
        }
        return repository.list(limit, offset);
    }

    public long count() {
        return repository.count();
    }

    public DocumentRecord findDocument(String id) {
        return repository.findDocument(id)
                .orElseThrow(() -> new ResourceNotFoundException("Document not found"));
    }

    public TaskRecord findTask(String id) {
        return repository.findTask(id)
                .orElseThrow(() -> new ResourceNotFoundException("Task not found"));
    }

    public List<TaskRecord> findTasksByDocument(String documentId) {
        findDocument(documentId);
        return repository.findTasksByDocument(documentId);
    }

    private void removeAfterFailure(String objectKey, RuntimeException failure) {
        try {
            storage.remove(objectKey);
        } catch (RuntimeException cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
        }
    }

    private void validatePdf(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("PDF file is required");
        }
        try (InputStream input = file.getInputStream()) {
            if (!"%PDF-".equals(new String(input.readNBytes(5), java.nio.charset.StandardCharsets.US_ASCII))) {
                throw new IllegalArgumentException("Uploaded file is not a PDF");
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Could not read uploaded PDF", e);
        }
    }

    private String normalizeCollection(String value) {
        String collection = value == null ? "default" : value.trim();
        if (!collection.matches("[A-Za-z0-9_-]{1,100}")) {
            throw new IllegalArgumentException("collection must be 1..100 letters, digits, _ or -");
        }
        return collection;
    }

    private String normalizeFileName(String value) {
        String normalized = value == null ? "document.pdf" : value.replace('\\', '/');
        String name = normalized.substring(normalized.lastIndexOf('/') + 1).trim();
        if (name.isEmpty() || name.length() > 255 || !name.toLowerCase(java.util.Locale.ROOT).endsWith(".pdf")) {
            throw new IllegalArgumentException("file name must be a PDF name of at most 255 characters");
        }
        return name;
    }

    public record UploadResult(String documentId, String taskId,
                               String documentStatus, String taskStatus) {
    }
}
