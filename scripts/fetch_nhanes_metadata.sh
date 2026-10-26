#!/usr/bin/env bash
# Klonuje repo ccb-hms/NHANES-metadata (mapowania ontologiczne + tabele grafu
# ontologii) do external/NHANES-metadata. Nie jest commitowane do tego repo
# (ma ~270MB) -- uruchom to raz po sklonowaniu.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d external/NHANES-metadata ]; then
  echo "external/NHANES-metadata already present"
  exit 0
fi

mkdir -p external
git clone --depth 1 https://github.com/ccb-hms/NHANES-metadata.git external/NHANES-metadata
