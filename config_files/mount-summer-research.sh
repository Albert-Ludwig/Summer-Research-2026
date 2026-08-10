#!/usr/bin/env bash

set -e

SOURCE="/mnt/c/Users/Administrator/Documents/Summer Research 2026"
TARGET="/srv/summer-research-2026"

mkdir -p "$TARGET"

if ! mountpoint -q "$TARGET"; then
  mount --bind "$SOURCE" "$TARGET"
fi
