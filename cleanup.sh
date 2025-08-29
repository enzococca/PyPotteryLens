#!/bin/sh

# Exit on error
set -e

printf "================================================================================\n"
printf "                    PyPotteryLens Environment Cleanup\n"
printf "================================================================================\n\n"

printf "[!] This will remove the virtual environment and require reinstallation.\n"
printf "    Continue? (y/N): "
read -r response

if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
    printf "Cleanup cancelled.\n"
    exit 0
fi

printf "\n[*] Cleaning up virtual environment...\n"

# Remove virtual environment
if [ -d "venv" ]; then
    rm -rf venv
    printf "[✓] Virtual environment removed\n"
else
    printf "[!] No virtual environment found\n"
fi

# Remove __pycache__ directories
printf "[*] Removing Python cache files...\n"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
printf "[✓] Cache files removed\n"

# Remove .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null || true
printf "[✓] Compiled Python files removed\n"

printf "\n[✓] Cleanup complete!\n"
printf "\nTo reinstall PyPotteryLens, run:\n"
printf "    sh PyPotteryLens_UNIX.sh\n\n"