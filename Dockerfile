# Use Python 3.9 slim as the base image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app/src

# Copy the content of the current directory to /app/src
COPY . /app

# Ensure that the data file is copied into the container
COPY ./data /app/data

# Install required packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port that the Flask app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
