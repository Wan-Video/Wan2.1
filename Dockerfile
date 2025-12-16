# Use a PyTorch base image with CUDA support
# Check NVIDIA's NGC catalog or Docker Hub for suitable images
# Example: nvcr.io/nvidia/pytorch:24.03-py3 (PyTorch 2.4.0, Python 3.10, CUDA 12.1)
# Ensure the Python version in the base image is compatible with project dependencies.
# Python 3.10 is generally a safe bet for recent PyTorch versions.
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set the working directory
WORKDIR /app

# Install essential build tools and Python headers (if needed for some packages)
# RUN apt-get update && apt-get install -y --no-install-recommends #     build-essential #     python3-dev #  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Upgrading pip, setuptools, and wheel first can prevent some build issues.
RUN pip install --upgrade pip setuptools wheel

# Special handling for flash-attn as per INSTALL.md
# Attempting the no-build-isolation method first.
# Ensure that the PyTorch and CUDA versions are compatible with flash-attn
RUN pip install flash-attn --no-build-isolation

# Install other dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make generate.py executable (if it's intended to be run directly and has a shebang)
# RUN chmod +x generate.py

# Set environment variables (optional, can be overridden at runtime)
# Example: Set a default Gradio app to run
ENV GRADIO_APP_SCRIPT gradio/t2v_14B_singleGPU.py
# ENV MODEL_CKPT_DIR ./Wan2.1-T2V-14B # Or a path inside the container where models will be mounted/downloaded

# Expose the default Gradio port (usually 7860)
EXPOSE 7860

# Default command to run the Gradio application
# This assumes the Gradio apps are launched with `python <script_name>`
# Users might need to adjust this based on how they want to run their app.
# Using `sh -c` to allow environment variable substitution in the command.
CMD ["sh", "-c", "python $GRADIO_APP_SCRIPT"]
