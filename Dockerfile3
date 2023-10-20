# Use the specified PyTorch base image
FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04 as runtime

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies using pip
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the model files
COPY src/readability_classifier /app/src/readability_classifier

# Add the project root directory to the Python path
ENV PYTHONPATH /app:/app/src:/app/src/readability_classifier:$PYTHONPATH

# Define the command to run when the container starts
CMD ["python", "src/readability_classifier/main.py", "-h"]
