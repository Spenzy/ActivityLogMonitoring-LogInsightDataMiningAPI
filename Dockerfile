# Use the official Python image as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files to the working directory
COPY . .

# Expose the port on which your Flask app runs (default is 5000)
EXPOSE 5000

# Set the environment variables (optional)
# ENV FLASK_ENV=production
ENV FLASK_APP=server.py

# Run the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
