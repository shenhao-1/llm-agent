package com.modularrag.documentapi.ingestion;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

@Configuration
public class KafkaTopicConfiguration {
    @Bean
    NewTopic ingestionTopic(IngestionProperties properties) {
        return TopicBuilder.name(properties.topic()).partitions(1).replicas(1).build();
    }
}
