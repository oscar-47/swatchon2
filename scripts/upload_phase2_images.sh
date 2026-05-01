#!/bin/bash
# Upload Phase 2 images to Google Drive
# Professional sources only (skip pexels/unsplash)

set -e

# Class folder IDs on Drive
declare -A CLASS_IDS
CLASS_IDS[Double_Jersey]="1qEcwTnzCT2zvGaYrIeLtnnBTQ2TcVBPO"
CLASS_IDS[Cable_Knit]="1l6o9q7lfeIwonwGpfSZwg_5nrwKq7WDf"
CLASS_IDS[Purl_Knit]="1zmZAaXnuCfUmTMXZkI2-EXG3lJZeK_ks"
CLASS_IDS[Intarsia]="1oPS7yMN--5rbeH5Llo_kHFlOCBlhSYXu"
CLASS_IDS[Raschel]="1x0usTsjSQ5pbont978ovIJGYTHL19yax"
CLASS_IDS[Basket_Hopsack]="168nChrHc9ebCg_otlHH9recCamOE3SYc"

# L1 mapping
declare -A L1_MAP
L1_MAP[Double_Jersey]="KNIT"
L1_MAP[Cable_Knit]="KNIT"
L1_MAP[Purl_Knit]="KNIT"
L1_MAP[Intarsia]="KNIT"
L1_MAP[Raschel]="KNIT"
L1_MAP[Basket_Hopsack]="WOVEN"

SKIP_SOURCES="pexels unsplash"
DATASET="FabricFlow_Dataset"

for class in Double_Jersey Cable_Knit Purl_Knit Intarsia Raschel Basket_Hopsack; do
    l1=${L1_MAP[$class]}
    class_drive_id=${CLASS_IDS[$class]}
    class_dir="$DATASET/$l1/$class"

    echo ""
    echo "=========================================="
    echo "  $class ($l1)"
    echo "=========================================="

    for source_dir in "$class_dir"/*/; do
        [ -d "$source_dir" ] || continue
        source=$(basename "$source_dir")

        # Skip stock photo sources
        skip=false
        for s in $SKIP_SOURCES; do
            [ "$source" = "$s" ] && skip=true
        done
        $skip && echo "  [skip] $source (stock photo)" && continue

        # Count JPGs
        count=$(find "$source_dir" -maxdepth 1 -name "*.jpg" | wc -l | tr -d ' ')
        [ "$count" -eq 0 ] && continue

        echo ""
        echo "  --- $source ($count images) ---"

        # Create source subfolder on Drive
        result=$(gws drive files create \
          --json "{\"name\": \"$source\", \"mimeType\": \"application/vnd.google-apps.folder\", \"parents\": [\"$class_drive_id\"]}" \
          --params '{"supportsAllDrives": true}' 2>&1)
        source_drive_id=$(echo "$result" | grep '"id"' | head -1 | sed 's/.*"id": "//;s/".*//')
        echo "  Drive folder: $source_drive_id"

        # Upload all JPGs
        uploaded=0
        for jpg in "$source_dir"*.jpg; do
            [ -f "$jpg" ] || continue
            gws drive files create \
              --json "{\"name\": \"$(basename "$jpg")\", \"parents\": [\"$source_drive_id\"]}" \
              --upload "$jpg" \
              --params '{"supportsAllDrives": true}' > /dev/null 2>&1
            uploaded=$((uploaded + 1))
            # Progress every 50
            if [ $((uploaded % 50)) -eq 0 ]; then
                echo "    uploaded $uploaded / $count"
            fi
        done
        echo "  ✓ $uploaded / $count uploaded"
    done
done

echo ""
echo "=========================================="
echo "  ALL DONE"
echo "=========================================="
