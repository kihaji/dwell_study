# One image for the Spark master, workers and driver: driver and executors must run the
# same Python and library versions.
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Hadoop needs a named user; match the host uid so written results are owned by you
ARG UID=1000
RUN useradd -m -u ${UID} dwells
USER dwells

# Code is mounted at /work (no rebuild after code changes)
ENV PYTHONPATH=/work/src:/work/spark \
    SPARK_HOME=/usr/local/lib/python3.12/site-packages/pyspark \
    PYSPARK_PYTHON=python3 \
    NUMBA_CACHE_DIR=/tmp/numba \
    SPARK_WORKER_DIR=/tmp/spark-work
WORKDIR /work
