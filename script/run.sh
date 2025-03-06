#!/bin/bash

echo "Starting Spark jobs..."

echo "Running process_data..."
# Run the process_data function
spark-submit --master spark://spark-master:7077 /code/src/run.py process_data -cfg /code/config/cfg.yaml -dataset news -dirout /data/

echo "Running process_data_all..."
# Run the process_data_all function
spark-submit --master spark://spark-master:7077 /code/src/run.py process_data_all -cfg /code/config/cfg.yaml -dataset news -dirout /data/

echo "Spark job completed."
