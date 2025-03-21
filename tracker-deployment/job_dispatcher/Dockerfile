FROM tracker-deployment-base

WORKDIR /app

COPY job_dispatcher.py /app/

# Run dispatcher automatically on startup
ENTRYPOINT ["python", "/app/job_dispatcher.py"]
