#!/usr/bin/env bash
# DriftTrace container entrypoint: selects the role from the first argument.
#   api      -> FastAPI prediction service (uvicorn)
#   webhook  -> local webhook stub receiver (uvicorn)
#   monitor  -> streaming monitor (Redpanda or file source)
#   cli      -> pass remaining args straight to the drifttrace CLI
set -euo pipefail

ROLE="${1:-api}"
shift || true

case "${ROLE}" in
  api)
    exec uvicorn "drifttrace.serving.app:create_app" \
      --factory --host 0.0.0.0 --port "${DRIFTTRACE_API_PORT:-8000}"
    ;;
  webhook)
    exec uvicorn "drifttrace.alerting.webhook_stub:create_app" \
      --factory --host 0.0.0.0 --port "${DRIFTTRACE_WEBHOOK_PORT:-9000}"
    ;;
  monitor)
    # Bounded monitor run over Redpanda; args allow overriding brokers/topic/etc.
    exec python -m drifttrace.cli.main monitor \
      --brokers "${REDPANDA_BROKER:-redpanda:9092}" \
      --topic "${DRIFTTRACE_TOPIC:-drifttrace.predictions}" \
      --webhook-url "${WEBHOOK_STUB_URL:-http://webhook-stub:9000/alert}" \
      "$@"
    ;;
  cli)
    exec python -m drifttrace.cli.main "$@"
    ;;
  *)
    echo "unknown role: ${ROLE} (expected api|webhook|monitor|cli)" >&2
    exit 64
    ;;
esac
