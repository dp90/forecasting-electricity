# Base image with PyTorch, Python, and CUDA
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy your application's files into the container
COPY . .

# Install additional dependencies if there are any
RUN pip install -r requirements.txt

# Command to run the application
CMD ["python", "nanogpt/bigram.py"]