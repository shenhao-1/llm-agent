FROM alpine:3.22 AS fetch
RUN apk add --no-cache ca-certificates curl
ARG MINIO_RELEASE=RELEASE.2025-09-07T16-13-09Z
ARG MINIO_SHA256=7c5bd8512c6e966455b1d198209358b2d191c77a83ab377c4073281065fb855f
RUN curl -fsSL "https://github.com/minio/minio/releases/download/${MINIO_RELEASE}/minio.linux-amd64.${MINIO_RELEASE}" -o /usr/local/bin/minio \
    && echo "${MINIO_SHA256}  /usr/local/bin/minio" | sha256sum -c - \
    && chmod +x /usr/local/bin/minio

FROM alpine:3.22
RUN apk add --no-cache ca-certificates
COPY --from=fetch /usr/local/bin/minio /usr/local/bin/minio
ENTRYPOINT ["/usr/local/bin/minio"]
