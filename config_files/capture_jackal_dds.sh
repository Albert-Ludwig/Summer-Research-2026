#!/usr/bin/env bash

set -e

CAPTURE_FILE="/tmp/jackal_dds_capture.txt"
CONTAINER_NAME="wsl_humble_jackal_discovery"

timeout 22 tcpdump -ni eth0 -vv "host 192.168.131.1" > "$CAPTURE_FILE" 2>&1 &
CAPTURE_PID=$!

sleep 1
docker start -a "$CONTAINER_NAME" || true
wait "$CAPTURE_PID" || true

echo "=== Jackal DDS UDP capture ==="
cat "$CAPTURE_FILE"
