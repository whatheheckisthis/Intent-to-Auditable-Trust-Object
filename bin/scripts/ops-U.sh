#!/bin/bash

# Main menu script to run utility operations

while true; do
    clear
    echo "====== Ops Utilities Menu ======"
    echo "1. Cleanup Logs"
    echo "2. Backup GitHub Repo"
    echo "3. Check Python Virtual Environment"
    echo "4. Tail Log with Timestamps"
    echo "5. Convert Markdown to HTML"
    echo "6. Initialize Environment"
    echo "7. Install Zsh and Homebrew"
    echo "8. Exit"
    echo "================================"
    read -p "Select an option [1-8]: " option

    case $option in
        1) bash cleanup_logs.sh ;;
        2) bash backup_repo.sh ;;
        3) bash check_venv.sh ;;
        4) bash timestamp_tail.sh ;;
        5) bash md_to_html.sh ;;
        6) bash init_environment.sh ;;
        7) bash setup_zsh_homebrew.sh ;;
        8) echo "Exiting..."; break ;;
        *) echo "Invalid option. Please try again." ;;
    esac

    read -p "Press [Enter] key to continue..." readEnterKey
done
