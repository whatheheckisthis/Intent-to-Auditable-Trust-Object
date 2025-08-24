#!/bin/bash

# ops.sh - Central menu for operations utilities
# Usage: ./ops.sh

show_menu() {
  echo "==========================="
  echo "     OPS UTILITIES MENU    "
  echo "==========================="
  echo "1) Clean up stale logs"
  echo "2) Backup GitHub repository"
  echo "3) Check Python virtual environment"
  echo "4) Add timestamps to live logs"
  echo "5) Convert Markdown to HTML"
  echo "6) Initialize development environment"
  echo "7) Exit"
  echo ""
}

run_choice() {
  case $1 in
    1) bash cleanup_logs.sh ;;
    2) bash backup_repo.sh ;;
    3) bash check_venv.sh ;;
    4) bash timestamp_tail.sh ;;
    5) bash md_to_html.sh ;;
    6) bash init_environment.sh ;;
    7) echo "Exiting. Goodbye!" ; exit 0 ;;
    *) echo "Invalid option. Try again." ;;
  esac
}

while true; do
  show_menu
  read -p "Choose an option [1-7]: " choice
  run_choice $choice
  echo ""
done
