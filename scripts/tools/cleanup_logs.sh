#!/bin/bash
#
# scripts/cleanup_logs.sh
#
# Purpose:
#   Archive and clean up log files to maintain a tidy workspace.
#
# Description:
#   This script moves log files older than a specified number of days
#   from the main logs directory into an archive folder, and deletes
#   archived logs older than 30 days.
#
# Usage:
#   Run this script manually or integrate it into automated maintenance workflows.
#
# Requirements:
#   - Bash shell
#   - Standard Unix utilities: mkdir, find, mv, rm
#
# Notes:
#   Customize DAYS_TO_KEEP as needed to adjust log retention period.
#

LOG_DIR="./logs"
ARCHIVE_DIR="./logs/archive"
DAYS_TO_KEEP=7

echo "Starting log cleanup..."

mkdir -p "$ARCHIVE_DIR"

find "$LOG_DIR" -type f -name "*.log" -mtime +$DAYS_TO_KEEP -exec mv {} "$ARCHIVE_DIR" \;

find "$ARCHIVE_DIR" -type f -name "*.log" -mtime +30 -delete

echo "Log cleanup complete. Archived logs older than $DAYS_TO_KEEP days."

# End of script

