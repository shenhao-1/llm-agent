package com.modularrag.documentapi.storage;

import java.io.InputStream;

public interface ObjectStorage {
    void put(String objectKey, InputStream input, long size);

    void remove(String objectKey);
}
