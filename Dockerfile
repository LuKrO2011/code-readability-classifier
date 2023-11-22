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

# Copy the source code, resources (model and datasets) and scripts
COPY src /app/src
COPY res /app/res
COPY scripts /app/scripts

# Give write permission to the user
RUN chmod -R 777 /app

# Add the project root directory to the Python path
ENV PYTHONPATH /app:/app/src:/app/src/readability_classifier:$PYTHONPATH

# Specify volumes (res, scripts)
VOLUME /app/res
VOLUME /app/scripts

# Define the command to run when the container starts
#CMD ["python", "src/readability_classifier/main.py", "-h"]
#CMD ["python", "src/readability_classifier/main.py", "TRAIN", "-i", "res/datasets/combined", "--intermediate", "res/datasets/encoded", "-s", "res/models", "-k", "10"]
CMD ["sh"]

# Version tag
LABEL version="2"
