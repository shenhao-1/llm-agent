package com.modularrag.documentapi.ingestion;

import java.time.Instant;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

@Service
public class IngestionService {
    private final IngestionRepository repository;
    private final PythonIngestionClient python;
    private final IngestionProperties properties;
    private final TransactionTemplate transactions;

    public IngestionService(IngestionRepository repository, PythonIngestionClient python,
                            IngestionProperties properties, TransactionTemplate transactions) {
        this.repository = repository;
        this.python = python;
        this.properties = properties;
        this.transactions = transactions;
    }

    public void process(String taskId, int attemptNo) {
        Instant now = Instant.now();
        IngestionRepository.WorkItem item = transactions.execute(status -> repository.claim(
                taskId, attemptNo, now, now.plusSeconds(properties.leaseSeconds())));
        if (item == null) {
            return; // Kafka can redeliver a published event; only the matching attempt may run.
        }
        try {
            python.ingest(item);
        } catch (RuntimeException failure) {
            String reason = failure.getMessage();
            if (reason == null) {
                reason = failure.getClass().getSimpleName();
            }
            final String bounded = reason.substring(0, Math.min(reason.length(), 1000));
            transactions.executeWithoutResult(status -> repository.fail(
                    item, bounded, Instant.now(), properties.maxAttempts()));
            return;
        }
        transactions.executeWithoutResult(status -> repository.complete(item, Instant.now()));
    }

    public void recoverExpired() {
        for (IngestionRepository.WorkItem item : repository.expired(Instant.now())) {
            transactions.executeWithoutResult(status -> repository.recover(
                    item, Instant.now(), properties.maxAttempts()));
        }
    }
}
