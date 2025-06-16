FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
