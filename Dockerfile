# Stage 1: Build Stage
# Use the specified PyTorch base image
FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Copy the entire readability_classifier directory into the container
COPY src/readability_classifier /app/readability_classifier

# Copy your poetry files into the container
COPY pyproject.toml poetry.lock /app/

# Install Python and required system packages as root
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
USER user

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python -

# Set the PATH environment variable
ENV PATH="/home/user/.local/bin:$PATH"

# Install Python dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install

# Define the command to run when the container starts
CMD ["python", "readability_classifier/main.py -h"]
