# Use an official Python runtime as a base image
FROM python:3.9-slim
ENV PYTHONUNBUFFERED=TRUE
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY requirements_deployment.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_deployment.txt

COPY ["py_scripts/__init__.py", "py_scripts/model_interface.py", "./py_scripts/"]
COPY ["data/train_answers.csv", "data/test_answers.csv", "./data/"]
COPY ["model.h5", "py_scripts/api_server.py", "/"]
# Expose port 5678 for Gunicorn
EXPOSE 5678

# Define environment variables
ENV FLASK_APP=api_server.py

# Command to run on container start
CMD ["gunicorn", "--bind", "0.0.0.0:5678", "api_server:app"]
