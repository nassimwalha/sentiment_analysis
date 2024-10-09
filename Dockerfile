FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install any necessary packages specified in requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . /app


# Expose port 5000 (Flask default)
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
