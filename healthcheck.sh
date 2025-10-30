#!/usr/bin/env bash

URL="http://127.0.0.1:8000/health"

status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$URL") || status=0

if [ "$status" = "200" ]; then
  echo "✅ Healthcheck passed"
  exit 0
else
  echo "❌ Healthcheck failed with status $status"
  exit 1
fi
