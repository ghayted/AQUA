FROM python:3.9-slim

WORKDIR /app

# Force Python to flush stdout/stderr immediately
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stmodel.py .
COPY generate_historical_data.py .
COPY trained_weights.pth .

CMD ["python", "-u", "stmodel.py"]


