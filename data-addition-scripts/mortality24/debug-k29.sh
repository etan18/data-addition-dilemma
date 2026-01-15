#!/bin/bash

hospital_ids=()
while IFS= read -r line; do
    hospital_ids+=("$line")
done < /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu/above2000.txt

# Slice the array to get the first 10 hospital IDs
hospital_ids_subset=("${hospital_ids[@]}")

printf -v joined '%s-' "${hospital_ids_subset[@]}"
joined="${joined%-}"

PATH="$PWD/.venv/bin:$PATH" uv run --python 3.11 python -m icu_benchmarks.run \
    -d /home/tane/YAIB/YAIB-cohorts/data/mortality24/eicu \
    -n eicu \
    -t BinaryClassification \
    -tn Mortality24 \
    -hi "$joined" \
    -hit "338" \
    --complete-train \
    -m K29 \
    --cpu \
    -ls _catboost