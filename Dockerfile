FROM bitnami/spark:3.5.1

# Switch to root to install system dependencies
USER root

# Update system and install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir pyspark numpy pandas scipy scikit-learn pyarrow
RUN pip3 install --no-cache-dir awswrangler polars orjson duckdb s3fs smart-open
RUN pip3 install --no-cache-dir onnxruntime spacy seqeval gensim numba sqlalchemy pytest
RUN pip3 install --no-cache-dir transformers accelerate 
RUN pip3 install --no-cache-dir datasets

# Switch back to the Spark user for security
# USER 1001

# Set working directory
WORKDIR /code

# Copy source code
COPY code/ /code/
COPY script/run.sh /script/

# Set permissions for execution
RUN chmod +x /script/run.sh

# Default command to run Spark job
CMD ["/bin/bash", "/script/run.sh"]
