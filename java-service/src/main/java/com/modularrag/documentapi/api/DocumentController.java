package com.modularrag.documentapi.api;

import com.modularrag.documentapi.document.DocumentRecord;
import com.modularrag.documentapi.document.DocumentService;
import com.modularrag.documentapi.document.TaskRecord;
import java.net.URI;
import java.time.Instant;
import java.util.List;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api")
public class DocumentController {
    private final DocumentService service;

    public DocumentController(DocumentService service) {
        this.service = service;
    }

    @PostMapping(path = "/documents", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<DocumentService.UploadResult> upload(
            @RequestPart("file") MultipartFile file,
            @RequestParam(defaultValue = "default") String collection) {
        var result = service.upload(file, collection);
        return ResponseEntity.created(URI.create("/api/documents/" + result.documentId()))
                .body(result);
    }

    @GetMapping("/documents")
    public DocumentPage list(@RequestParam(defaultValue = "20") int limit,
                             @RequestParam(defaultValue = "0") int offset) {
        List<DocumentView> items = service.list(limit, offset).stream()
                .map(DocumentView::from).toList();
        return new DocumentPage(items, service.count(), limit, offset);
    }

    @GetMapping("/documents/{id}")
    public DocumentView getDocument(@PathVariable String id) {
        return DocumentView.from(service.findDocument(id));
    }

    @GetMapping("/tasks/{id}")
    public TaskRecord getTask(@PathVariable String id) {
        return service.findTask(id);
    }

    @GetMapping("/documents/{id}/tasks")
    public List<TaskRecord> getDocumentTasks(@PathVariable String id) {
        return service.findTasksByDocument(id);
    }

    public record DocumentPage(List<DocumentView> items, long total, int limit, int offset) {
    }

    // Internal MinIO object keys never appear in the public document response.
    public record DocumentView(String id, String collection, String fileName,
                               long sizeBytes, String status, Instant createdAt, Instant updatedAt) {
        static DocumentView from(DocumentRecord record) {
            return new DocumentView(record.id(), record.collection(), record.fileName(),
                    record.sizeBytes(), record.status(), record.createdAt(), record.updatedAt());
        }
    }
}
