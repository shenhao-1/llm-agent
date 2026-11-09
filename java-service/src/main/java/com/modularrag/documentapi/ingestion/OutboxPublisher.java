package com.modularrag.documentapi.ingestion;

import java.time.Instant;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class OutboxPublisher {
    private static final Logger log = LoggerFactory.getLogger(OutboxPublisher.class);
    private final IngestionRepository repository;
    private final KafkaTemplate<String, String> kafka;
    private final IngestionProperties properties;
    private final IngestionService ingestion;

    public OutboxPublisher(IngestionRepository repository, KafkaTemplate<String, String> kafka,
                           IngestionProperties properties, IngestionService ingestion) {
        this.repository = repository;
        this.kafka = kafka;
        this.properties = properties;
        this.ingestion = ingestion;
    }

    @Scheduled(fixedDelay = 5000)
    public void publish() {
        for (IngestionRepository.OutboxEvent event : repository.dueEvents(Instant.now())) {
            try {
                kafka.send(properties.topic(), event.taskId(),
                        event.taskId() + "|" + event.attemptNo()).get(5, TimeUnit.SECONDS);
                repository.markPublished(event.id(), Instant.now());
            } catch (Exception e) {
                if (e instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                }
                log.warn("Kafka publish failed for task {}: {}", event.taskId(), e.toString());
                return; // The outbox row stays pending until Kafka is available.
            }
        }
    }

    @Scheduled(fixedDelay = 30000)
    public void recover() {
        ingestion.recoverExpired();
    }
}
