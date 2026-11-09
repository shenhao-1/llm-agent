package com.modularrag.documentapi.ingestion;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.net.URI;
import java.time.Instant;
import java.util.function.Consumer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.transaction.TransactionStatus;
import org.springframework.transaction.support.TransactionCallback;
import org.springframework.transaction.support.TransactionTemplate;

class IngestionServiceTest {
    private final IngestionRepository repository = mock(IngestionRepository.class);
    private final PythonIngestionClient python = mock(PythonIngestionClient.class);
    private final TransactionTemplate transactions = mock(TransactionTemplate.class);
    private final IngestionProperties properties = new IngestionProperties("topic", URI.create("http://localhost"),
            "token", 3, 600, 300);
    private final IngestionRepository.WorkItem item = new IngestionRepository.WorkItem(
            "task", "document", "docs", "documents/document/original.pdf", 1);
    private IngestionService service;

    @BeforeEach
    void setUp() {
        service = new IngestionService(repository, python, properties, transactions);
        doAnswer(invocation -> {
            TransactionCallback<?> callback = invocation.getArgument(0);
            return callback.doInTransaction(mock(TransactionStatus.class));
        }).when(transactions).execute(any());
        doAnswer(invocation -> {
            Consumer<TransactionStatus> callback = invocation.getArgument(0);
            callback.accept(mock(TransactionStatus.class));
            return null;
        }).when(transactions).executeWithoutResult(any());
    }

    @Test
    void duplicateAttemptDoesNotCallPython() {
        service.process("task", 0);
        verify(python, never()).ingest(any());
    }

    @Test
    void successfulAttemptCompletesTask() {
        when(repository.claim(eq("task"), eq(0), any(Instant.class), any(Instant.class)))
                .thenReturn(item);
        service.process("task", 0);
        verify(python).ingest(item);
        verify(repository).complete(eq(item), any(Instant.class));
    }

    @Test
    void failedAttemptRecordsRetryState() {
        when(repository.claim(eq("task"), eq(0), any(Instant.class), any(Instant.class)))
                .thenReturn(item);
        doThrow(new IllegalStateException("Python unavailable")).when(python).ingest(item);
        service.process("task", 0);
        verify(repository).fail(eq(item), eq("Python unavailable"), any(Instant.class), eq(3));
    }
}
