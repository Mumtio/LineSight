# Windows equivalent of fetch_data.sh. Same two datasets, same deliberate
# omission of MVTec AD (Kaggle-only, D3).
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force data | Out-Null
kaggle datasets download -d nexuswho/aitex-fabric-image-database -p data/aitex --unzip
kaggle datasets download -d priemshpathirana/fabric-stain-dataset -p data/fabric_stain --unzip
Write-Output "Local datasets ready. Total footprint < 250 MB."
