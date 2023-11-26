# Use the specified PyTorch base image
FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04 as runtime

# Give the user admin privileges
USER root

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies using pip
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Install wkhtmltopdf
RUN apt-get update
RUN echo "8" | DEBIAN_FRONTEND=noninteractive apt-get install -y wkhtmltopdf

# Copy the source code
COPY src /app/src

# Give write permission to the user
RUN chmod -R 777 /app

# Add the project root directory to the Python path
ENV PYTHONPATH /app:/app/src:/app/src/readability_classifier:$PYTHONPATH
