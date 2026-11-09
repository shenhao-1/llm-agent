package com.modularrag.documentapi.ingestion;

import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
public class IngestionConsumer {
    private final IngestionService ingestion;

    public IngestionConsumer(IngestionService ingestion) {
        this.ingestion = ingestion;
    }

    @KafkaListener(topics = "${app.ingestion.topic}")
    public void consume(String event) {
        String[] parts = event.split("\\|", -1);
        if (parts.length != 2 || !parts[0].matches("[0-9a-fA-F-]{36}")) {
            throw new IllegalArgumentException("Invalid ingestion event: " + event);
        }
        ingestion.process(parts[0], Integer.parseInt(parts[1]));
    }
}
