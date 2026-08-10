#!/usr/bin/env bash

echo "=== Sudo ==="
if sudo -n true 2>/dev/null; then
  echo "SUDO_NOPASSWD=YES"
else
  echo "SUDO_NOPASSWD=NO"
fi

echo "=== Discovery start script ==="
sed -n '1,240p' /etc/clearpath/discovery-server-start 2>&1 || true

echo "=== Clearpath discovery files ==="
find /etc/clearpath -maxdepth 4 -type f 2>/dev/null |
  grep -Ei 'discover|fastdds|ros.*env' |
  sort || true

echo "=== Discovery process command ==="
ps -eo pid,user,args |
  grep -E 'fastdds.py discovery|fast-discovery-server' |
  grep -v grep || true

echo "=== Discovery socket ==="
ss -lunp 2>/dev/null | grep ':11811' || ss -lun | grep ':11811' || true

echo "=== Fast DDS version ==="
fastdds --version 2>&1 || true
dpkg-query -W 'fastdds*' 'libfastrtps*' 'ros-humble-fastrtps*' 2>/dev/null || true

echo "=== Discovery service journal ==="
journalctl -u clearpath-discovery.service -b --no-pager -n 120 2>&1 || true
