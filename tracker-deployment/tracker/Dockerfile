FROM tracker-deployment-base

WORKDIR /app

COPY tracker.py /app/
COPY requirements.txt /app/

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/app/tracker.py"]
