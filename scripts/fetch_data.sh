#!/usr/bin/env bash
# Local datasets only. MVTec AD is deliberately NOT fetched here:
# it is mounted read-only inside a Kaggle session (see docs/datasets.md).
#
# Requires: pip install kaggle, and a token at ~/.kaggle/kaggle.json
# (Kaggle -> Settings -> API -> Create New Token), chmod 600.
set -euo pipefail
mkdir -p data
kaggle datasets download -d nexuswho/aitex-fabric-image-database -p data/aitex --unzip
kaggle datasets download -d priemshpathirana/fabric-stain-dataset -p data/fabric_stain --unzip
echo "Local datasets ready. Total footprint < 250 MB."
