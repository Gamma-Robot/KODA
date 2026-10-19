#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility entry. The root-level batch script owns the production
# contract: robot body and contact-aware Dex3 hand both come from tracking.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${PROJECT_ROOT}/run_batch_singleframe_view_change_30to50.sh" "$@"
