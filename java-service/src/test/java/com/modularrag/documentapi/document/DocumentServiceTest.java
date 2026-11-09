package com.modularrag.documentapi.document;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import com.modularrag.documentapi.storage.ObjectStorage;
import java.nio.charset.StandardCharsets;
import java.util.function.Consumer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.dao.DataAccessResourceFailureException;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.transaction.TransactionStatus;
import org.springframework.transaction.support.TransactionTemplate;

class DocumentServiceTest {
    private final DocumentRepository repository = mock(DocumentRepository.class);
    private final ObjectStorage storage = mock(ObjectStorage.class);
    private final TransactionTemplate transactions = mock(TransactionTemplate.class);
    private DocumentService service;

    @BeforeEach
    void setUp() {
        service = new DocumentService(repository, storage, transactions);
        doAnswer(invocation -> {
            Consumer<TransactionStatus> action = invocation.getArgument(0);
            action.accept(mock(TransactionStatus.class));
            return null;
        }).when(transactions).executeWithoutResult(any());
    }

    @Test
    void uploadStoresPdfAndCreatesPendingTask() {
        var file = pdf("report.pdf");

        var result = service.upload(file, "docs");

        assertEquals("UPLOADED", result.documentStatus());
        assertEquals("PENDING", result.taskStatus());
        verify(storage).put(anyString(), any(), org.mockito.ArgumentMatchers.eq(file.getSize()));
        verify(repository).insert(any(DocumentRecord.class), any(TaskRecord.class));
    }

    @Test
    void databaseFailureRemovesPreviouslyUploadedObject() {
        doThrow(new DataAccessResourceFailureException("database down"))
                .when(repository).insert(any(DocumentRecord.class), any(TaskRecord.class));

        assertThrows(DataAccessResourceFailureException.class,
                () -> service.upload(pdf("report.pdf"), "docs"));

        verify(storage).remove(anyString());
    }

    @Test
    void rejectsNonPdfBeforeWriting() {
        var file = new MockMultipartFile("file", "report.pdf", "application/pdf",
                "this is not a PDF".getBytes(StandardCharsets.UTF_8));

        assertThrows(IllegalArgumentException.class, () -> service.upload(file, "docs"));

        verify(storage, never()).put(anyString(), any(), org.mockito.ArgumentMatchers.anyLong());
    }

    private MockMultipartFile pdf(String name) {
        return new MockMultipartFile("file", name, "application/pdf",
                "%PDF-1.7\nexample".getBytes(StandardCharsets.US_ASCII));
    }
}
