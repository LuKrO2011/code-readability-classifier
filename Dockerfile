# Use the specified PyTorch base image
FROM python:3.11-bookworm as runtime

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

# Add the project root directory to the Python path
ENV PYTHONPATH /app:/app/src:/app/src/readability_classifier:$PYTHONPATH
