#!/usr/bin/env bash
# Upload core_set knit images to Google Drive
# Only uploads confirmed mappings; skips lacoste, plain_pique, skip_stitch

set -eo pipefail

SRC="/Users/oscar/Downloads/fabricflow/swatchon2/coredataset"
DRIVE_PARAMS=',"includeItemsFromAllDrives":"true","supportsAllDrives":"true"'

# core_set folder IDs (inside the main class folders)
declare -A CORE_SET_IDS=(
  [Jersey]="1Uatlww007Odx2MZ09jaeVE5KjJEBnhSF"
  [Rib_Knit]="1WJ-oKC0cVtEgVCsesOsZ4OFjVgyIgvH8"
  [Double_Jersey]="1VRKJ6Ivp4mej3RMOq81cN050S1rzwfle"
  [Interlock]="1g6_fYQn1ii7V8kRpnF3wckmKt4yDDO5t"
  [Purl_Knit]="19NLG-9GwIp49MIn5m_5UzSEhElFeeozI"
  [French_Terry]="14jBhvYsYJZ1poN5c532xMK_zYSKv4nnF"
)

# Mapping: filename prefix → class
declare -A FILE_CLASS_MAP=(
  # Jersey
  [jersey]=Jersey
  # Rib_Knit
  ["1×1_rib_back"]=Rib_Knit
  ["1×1_rib_front"]=Rib_Knit
  ["2×2_rib_(2-in_1-out)_back"]=Rib_Knit
  ["2×2_rib_(2-in_1-out)_front"]=Rib_Knit
  ["2×2_rib_(2-in_2-out)_back"]=Rib_Knit
  ["2×2_rib_(2-in_2-out)_front"]=Rib_Knit
  [simple_rib]=Rib_Knit
  [board_rib]=Rib_Knit
  [half_cardigan]=Rib_Knit
  [full_cardigan]=Rib_Knit
  [ottoman]=Rib_Knit
  [racked_stitch]=Rib_Knit
  ["tubular_(split_welt)"]=Rib_Knit
  # Double_Jersey
  [double_jersey]=Double_Jersey
  [double_face]=Double_Jersey
  [ponte_di_roma]=Double_Jersey
  [double_pique]=Double_Jersey
  [full_milano]=Double_Jersey
  [half_milano]=Double_Jersey
  # Interlock
  [interlock]=Interlock
  # Purl_Knit
  [links]=Purl_Knit
  [seed_stitch]=Purl_Knit
  [double_seed_stitch]=Purl_Knit
  [moss_stitch]=Purl_Knit
  # French_Terry
  [plush]=French_Terry
)

uploaded=0
skipped=0
errors=0

for file in "$SRC"/*.jpeg; do
  fname=$(basename "$file")
  # Extract prefix: remove _base_core_set_Nx.jpeg
  prefix=$(echo "$fname" | sed 's/_base_core_set_[0-9]*x\.jpeg$//')

  class="${FILE_CLASS_MAP[$prefix]:-}"
  if [ -z "$class" ]; then
    echo "SKIP: $fname (no mapping for '$prefix')"
    ((skipped++))
    continue
  fi

  folder_id="${CORE_SET_IDS[$class]}"
  echo -n "UPLOAD: $fname → $class/core_set ... "

  if gws drive files create \
    --json "{\"name\":\"$fname\",\"parents\":[\"$folder_id\"]}" \
    --params '{"includeItemsFromAllDrives":"true","supportsAllDrives":"true"}' \
    --upload "$file" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id','FAIL'))" 2>/dev/null; then
    ((uploaded++))
  else
    echo "ERROR"
    ((errors++))
  fi
done

echo ""
echo "Done: $uploaded uploaded, $skipped skipped, $errors errors"
