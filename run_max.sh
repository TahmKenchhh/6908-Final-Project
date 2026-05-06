#!/bin/bash

cd "$(dirname "$0")"
source .venv/bin/activate
python main.py
status=$?

echo
echo "Max exited with status ${status}. Press Enter to close this window."
read -r
exit "$status"
