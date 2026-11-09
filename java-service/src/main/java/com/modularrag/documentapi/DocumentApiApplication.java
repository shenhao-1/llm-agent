package com.modularrag.documentapi;

import com.modularrag.documentapi.storage.MinioProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableScheduling;
import com.modularrag.documentapi.ingestion.IngestionProperties;

@SpringBootApplication
@EnableScheduling
@EnableConfigurationProperties({MinioProperties.class, IngestionProperties.class})
public class DocumentApiApplication {
    public static void main(String[] args) {
        SpringApplication.run(DocumentApiApplication.class, args);
    }
}
