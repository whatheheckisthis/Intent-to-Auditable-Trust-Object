# Run the script (moves the files)
bash archive_legacy.sh

# Check status to confirm moved files
git status

# Commit the move
git commit -m "Move legacy files to bin/legacy for archival"
