package com.modularrag.documentapi.ingestion;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;
import org.springframework.stereotype.Component;

@Component
public class PythonIngestionClient {
    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5)).build();
    private final ObjectMapper json;
    private final IngestionProperties properties;

    public PythonIngestionClient(ObjectMapper json, IngestionProperties properties) {
        this.json = json;
        this.properties = properties;
    }

    public void ingest(IngestionRepository.WorkItem item) {
        try {
            byte[] payload = json.writeValueAsBytes(Map.of(
                    "taskId", item.taskId(), "documentId", item.documentId(),
                    "collection", item.collection(), "objectKey", item.objectKey()));
            HttpRequest request = HttpRequest.newBuilder(properties.pythonUrl())
                    .timeout(Duration.ofSeconds(properties.requestTimeoutSeconds()))
                    .header("Content-Type", "application/json")
                    .header("X-Ingest-Token", properties.token())
                    .POST(HttpRequest.BodyPublishers.ofByteArray(payload)).build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                throw new IllegalStateException("Python ingest returned HTTP " + response.statusCode()
                        + ": " + response.body());
            }
        } catch (Exception e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new IllegalStateException("Python ingestion failed: " + e.getMessage(), e);
        }
    }
}
