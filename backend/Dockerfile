FROM python:3.14-alpine

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir websockets>=16.0

# Copy the server script into the container
COPY server.py .

# Expose the websocket port
EXPOSE 8765

# Run the server
CMD ["python", "server.py"]
