#!/usr/bin/env bash

set -e

: "${JACKAL_PASSWORD:?JACKAL_PASSWORD is required}"

export SSHPASS="$JACKAL_PASSWORD"

REMOTE_COMMAND="printf '%s\\n' '$JACKAL_PASSWORD' | sudo -S -p '' timeout 22 tcpdump -ni br0 -c 20 -vv udp port 11811"

sshpass -e ssh \
  -o ConnectTimeout=5 \
  administrator@192.168.131.1 \
  "$REMOTE_COMMAND"

unset SSHPASS
