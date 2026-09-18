FROM --platform=linux/amd64 python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

WORKDIR /workspace
COPY docker/offline-wheelhouse/build.sh /usr/local/bin/build-opd-wheelhouse
RUN chmod +x /usr/local/bin/build-opd-wheelhouse

ENTRYPOINT ["/usr/local/bin/build-opd-wheelhouse"]
