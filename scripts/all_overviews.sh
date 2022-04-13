#!/bin/bash

# Bash script for generating overviews for any combination of years and valid predicted forest structure metrics
# Modify lines 16,17 and 30,31 if needed. See 08-create-overviews.py for argument help.

declare -a yrarr=("2014")
declare -a regvararr=("canopy_cvr" "basal_area" "bio_acre")
 
# Run create-overviews for all yr regvar combinations
for year in ${yrarr[@]}; do
    for regvar in ${regvararr[@]}; do
        python 08-create-overviews.py \
            -z 6 \
            -m mosaicjson/$regvar-$year-z8-z14.json \
            -o overviews/$regvar/$year \
            -p "mosaicjson/$regvar-$year" \
            -c "https://usfs.blob.core.windows.net/app/overview_cog/$regvar/$year" \
            --min-zoom=6 \
            --max-zoom=7 \
            -t 1024
    done
done

#Run create-overviews for all yrs of classes
for year in ${yrarr[@]}; do
    python 08-create-overviews.py \
        -z 6 \
        -m mosaicjson/class-$year-z8-z14.json \
        -o overviews/class/$year \
        -p "mosaicjson/class-$year" \
        -c "https://usfs.blob.core.windows.net/app/overview_cog/class/$year" \
        --min-zoom=6 \
        --max-zoom=7 \
        -t 1024
done