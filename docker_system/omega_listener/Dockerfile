FROM python:3.11

WORKDIR /app

COPY omega_frame_listener.py /app/

# Install required Python packages
RUN pip install --no-cache-dir redis watchdog

# Ensure the script runs when the container starts
CMD ["python", "/app/omega_frame_listener.py"]
