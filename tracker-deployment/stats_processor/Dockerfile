FROM tracker-deployment-base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code into container
COPY . .

# Default command (can be overridden)
# CMD ["python", "compute_stats.py"]

