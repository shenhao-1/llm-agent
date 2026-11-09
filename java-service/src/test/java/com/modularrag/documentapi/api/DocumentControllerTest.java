package com.modularrag.documentapi.api;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.modularrag.documentapi.document.DocumentService;
import com.modularrag.documentapi.document.DocumentRecord;
import com.modularrag.documentapi.document.TaskRecord;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.multipart.MultipartFile;

class DocumentControllerTest {
    private final DocumentService service = mock(DocumentService.class);
    private MockMvc mvc;

    @BeforeEach
    void setUp() {
        mvc = MockMvcBuilders.standaloneSetup(new DocumentController(service))
                .setControllerAdvice(new ApiExceptionHandler())
                .build();
    }

    @Test
    void uploadReturnsDocumentAndTaskIdentifiers() throws Exception {
        when(service.upload(any(MultipartFile.class), eq("docs")))
                .thenReturn(new DocumentService.UploadResult("doc-1", "task-1", "UPLOADED", "PENDING"));
        var pdf = new MockMultipartFile("file", "report.pdf", "application/pdf",
                "%PDF-1.7".getBytes(StandardCharsets.US_ASCII));

        mvc.perform(multipart("/api/documents").file(pdf).param("collection", "docs"))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.documentId").value("doc-1"))
                .andExpect(jsonPath("$.taskId").value("task-1"))
                .andExpect(jsonPath("$.taskStatus").value("PENDING"));
    }

    @Test
    void taskEndpointReturnsStoredStatus() throws Exception {
        Instant now = Instant.parse("2026-01-01T00:00:00Z");
        when(service.findTask("task-1"))
                .thenReturn(new TaskRecord("task-1", "doc-1", "PENDING", 0, null, now, now));

        mvc.perform(get("/api/tasks/task-1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("PENDING"));
    }

    @Test
    void documentListDoesNotExposeMinioObjectKeys() throws Exception {
        Instant now = Instant.parse("2026-01-01T00:00:00Z");
        when(service.list(20, 0)).thenReturn(List.of(new DocumentRecord(
                "doc-1", "docs", "report.pdf", "documents/doc-1/original.pdf",
                "application/pdf", 123, "UPLOADED", now, now)));
        when(service.count()).thenReturn(1L);

        mvc.perform(get("/api/documents"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.total").value(1))
                .andExpect(jsonPath("$.items[0].fileName").value("report.pdf"))
                .andExpect(jsonPath("$.items[0].objectKey").doesNotExist());
    }
}
