#!/bin/bash
while true; do
    if ps aux | grep -q "[p]ython entrance.py"; then
        PID=$(ps aux | grep "[p]ython entrance.py" | awk '{print $2}')
        STATS=$(ps aux | grep "[p]ython entrance.py" | awk '{print "PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}')
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training running: $STATS"
        
        # Check for errors
        if tail -100 training_output.log 2>/dev/null | grep -qE "(Error|Traceback|Exception)"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR DETECTED!"
            tail -50 training_output.log | grep -E "(Error|Traceback|Exception)" | tail -5
        fi
        
        # Check for completion
        if tail -100 training_output.log 2>/dev/null | grep -qiE "(Done|Complete|Finished|Training completed)"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] TRAINING COMPLETED!"
            break
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training process not found"
        break
    fi
    sleep 60
done
