# This Dockerfile is optimised for AWS ECS using Python 3.11, and assumes CPU inference with OpenBLAS for local models.
# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.11.13-slim-bookworm AS builder

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

RUN pip install --no-cache-dir --target=/install torch==2.7.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
&& pip install --no-cache-dir --target=/install https://github.com/seanpedrick-case/llama-cpp-python-whl-builder/releases/download/v0.1.0/llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl \
&& pip install --no-cache-dir --target=/install -r requirements_no_local.txt

RUN rm requirements_no_local.txt

# Stage 2: Final runtime image
FROM public.ecr.aws/docker/library/python:3.11.13-slim-bookworm

# Install system dependencies.
RUN apt-get update \
    && apt-get clean \
    && apt-get install -y libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Make output folder
RUN mkdir -p /home/user/app/output \
&& mkdir -p /home/user/app/logs \
&& chown -R user:user /home/user/app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Switch to the "user" user
USER user

# Set environmental variables
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
	PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_SERVER_PORT=7860 \
	GRADIO_THEME=huggingface \
	SYSTEM=spaces
 
# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

CMD ["python", "app.py"]