# This Dockerfile is optimised for AWS ECS using Python 3.12, and assumes CPU inference with OpenBLAS for local models.
# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.12.11-slim-trixie AS builder

# Install system dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    #libopenblas-dev \
    pkg-config \
    python3-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements_no_local.txt .

# Set environment variables for OpenBLAS - not necessary if not building from source
# ENV OPENBLAS_VERBOSE=1
# ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

RUN pip install --no-cache-dir --target=/install torch==2.9.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
&& pip install --no-cache-dir --target=/install https://github.com/seanpedrick-case/llama-cpp-python-whl-builder/releases/download/v0.1.0/llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl \
&& pip install --no-cache-dir --target=/install -r requirements_lightweight.txt

RUN rm requirements_no_local.txt

# ===================================================================
# Stage 2: A common 'base' for both Lambda and Gradio
# ===================================================================
FROM public.ecr.aws/docker/library/python:3.12.11-slim-trixie AS base

# Set build-time and runtime environment variable for whether to run in Gradio mode or Lambda mode
ARG APP_MODE=gradio
ENV APP_MODE=${APP_MODE}

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV APP_HOME=/home/user

# Set env variables for Gradio & other apps
ENV GRADIO_TEMP_DIR=/tmp/gradio_tmp/ \
    MPLCONFIGDIR=/tmp/matplotlib_cache/ \
    GRADIO_OUTPUT_FOLDER=$APP_HOME/app/output/ \
    GRADIO_INPUT_FOLDER=$APP_HOME/app/input/ \
    FEEDBACK_LOGS_FOLDER=$APP_HOME/app/feedback/ \
    ACCESS_LOGS_FOLDER=$APP_HOME/app/logs/ \
    USAGE_LOGS_FOLDER=$APP_HOME/app/usage/ \
    CONFIG_FOLDER=$APP_HOME/app/config/ \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PATH=$APP_HOME/.local/bin:$PATH \
    PYTHONPATH=$APP_HOME/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces

# Copy Python packages from the builder stage
COPY --from=builder /install /usr/local/lib/python3.12/site-packages/
COPY --from=builder /install/bin /usr/local/bin/

# Copy your application code and entrypoint
COPY . ${APP_HOME}/app
COPY entrypoint.sh ${APP_HOME}/app/entrypoint.sh
# Fix line endings and set execute permissions
RUN sed -i 's/\r$//' ${APP_HOME}/app/entrypoint.sh \
    && chmod +x ${APP_HOME}/app/entrypoint.sh

WORKDIR ${APP_HOME}/app

# ===================================================================
# FINAL Stage 3: The Lambda Image (runs as root for simplicity)
# ===================================================================
FROM base AS lambda
# Set runtime ENV for Lambda mode
ENV APP_MODE=lambda
ENTRYPOINT ["/home/user/app/entrypoint.sh"]
CMD ["lambda_entrypoint.lambda_handler"]

# ===================================================================
# FINAL Stage 4: The Gradio Image (runs as a secure, non-root user)
# ===================================================================
FROM base AS gradio
# Set runtime ENV for Gradio mode
ENV APP_MODE=gradio

# Create non-root user
RUN useradd -m -u 1000 user

# Create the base application directory and set its ownership
RUN mkdir -p ${APP_HOME}/app && chown user:user ${APP_HOME}/app

# Create required sub-folders within the app directory and set their permissions
RUN mkdir -p \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config \
    && chown user:user \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config \
    && chmod 755 \
    ${APP_HOME}/app/output \
    ${APP_HOME}/app/input \
    ${APP_HOME}/app/logs \
    ${APP_HOME}/app/usage \
    ${APP_HOME}/app/feedback \
    ${APP_HOME}/app/config

# Now handle the /tmp directories
RUN mkdir -p /tmp/gradio_tmp /tmp/matplotlib_cache /tmp /var/tmp \
    && chown user:user /tmp /var/tmp /tmp/gradio_tmp /tmp/matplotlib_cache \
    && chmod 1777 /tmp /var/tmp /tmp/gradio_tmp /tmp/matplotlib_cache

# Fix apply user ownership to all files in the home directory
RUN chown -R user:user /home/user

# Set permissions for Python executable
RUN chmod 755 /usr/local/bin/python

# Declare volumes
VOLUME ["/tmp/matplotlib_cache"]
VOLUME ["/tmp/gradio_tmp"]
VOLUME ["/home/user/app/output"]
VOLUME ["/home/user/app/input"]
VOLUME ["/home/user/app/logs"]
VOLUME ["/home/user/app/usage"]
VOLUME ["/home/user/app/feedback"]
VOLUME ["/home/user/app/config"]
VOLUME ["/tmp"]
VOLUME ["/var/tmp"]

USER user

EXPOSE $GRADIO_SERVER_PORT

ENTRYPOINT ["/home/user/app/entrypoint.sh"]
CMD ["python", "app.py"]