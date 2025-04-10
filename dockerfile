   
   # Use an official Python runtime as a base image
   FROM python:3.9-slim

   # Install required system packages for OpenCV and GLib
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       libglib2.0-0 \
       && apt-get clean \
       && rm -rf /var/lib/apt/lists/*


   # Set the working directory inside the container
   WORKDIR /app

   # Copy the requirements.txt file to the container
   COPY requirements.txt /app/

   # Install any Python dependencies
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy the entire project to the container
   COPY . /app/

   # Expose port 5001 (default for Flask) and 4040 (for Ngrok)
   EXPOSE 8000


   # Set environment variables
   ENV FLASK_APP=app.py
   ENV FLASK_RUN_HOST=0.0.0.0

   # Entrypoint command
   CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
