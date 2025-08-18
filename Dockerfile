# Use an official Python 3.10 runtime as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Tell Docker that the container listens on port 7860 (standard for HF Spaces)
EXPOSE 7860

# Define the command to run the app when the container starts
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]