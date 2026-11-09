package com.modularrag.documentapi.ingestion;

import java.net.URI;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.ingestion")
public record IngestionProperties(String topic, URI pythonUrl, String token,
                                  int maxAttempts, int leaseSeconds, int requestTimeoutSeconds) {
}
