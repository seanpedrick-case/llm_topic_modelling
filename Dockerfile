# Stage 1: Build dependencies and download models
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm AS builder

# Install system dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    python3-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements_aws.txt .

RUN pip uninstall -y typing_extensions \
&& pip install --no-cache-dir --target=/install typing_extensions==4.12.2 \
&& pip install --no-cache-dir --target=/install torch==2.5.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
&& pip install --no-cache-dir --target=/install -r requirements_aws.txt

RUN rm requirements_aws.txt

# Stage 2: Final runtime image
FROM public.ecr.aws/docker/library/python:3.11.9-slim-bookworm

# Install system dependencies. Need to specify -y for poppler to get it to install
RUN apt-get update \
    && apt-get clean \
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
	TLDEXTRACT_CACHE=$HOME/app/tld/.tld_set_snapshot \
	SYSTEM=spaces
 
# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

CMD ["python", "app.py"]