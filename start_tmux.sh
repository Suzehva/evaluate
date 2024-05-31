#!/bin/bash

# Start a new tmux session named "my_session"
tmux new-session -d -s my_session

# Send the command to execute your shell script to the tmux session
tmux send-keys -t my_session "./run_overnight.sh" Enter

# Optionally, you can attach to the tmux session to view the output
tmux attach-session -t my_session
