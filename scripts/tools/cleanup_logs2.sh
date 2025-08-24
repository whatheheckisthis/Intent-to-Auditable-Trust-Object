#!/bin/bash
# Archive and clean old logs

mkdir -p logs/archive
find logs/ -name '*.log' -mtime +7 -exec mv {} logs/archive/ \;

