#!/bin/bash

while true; do
    conda activate honours_datapull
    current_time=$(date +"%H:%M")
    if [ "$current_time" == "17:30" ]; then
        python /path/to/your/python/file.py
    fi
    sleep 60
done