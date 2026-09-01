#!/bin/bash
# Monitor training progress

PID=$(ps aux | grep "peft_finetune.py" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "Training process not running"
    exit 1
fi

echo "Training Process PID: $PID"
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "CPU Usage:"
top -bn1 | grep "$PID" | awk '{print "CPU: " $9 "%", "MEM: " $10 "%" }'

echo ""
echo "Checkpoints:"
ls -lht outputs/ 2>/dev/null || echo "No checkpoints yet"

echo ""
echo "Latest output files:"
find . -name "*.log" -mmin -5 -exec tail -5 {} \; 2>/dev/null

echo ""
echo "Training progress: check for steps in outputs/ directory"