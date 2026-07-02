FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and outputs folder
COPY src/ src/
COPY config/ config/
COPY outputs/ outputs/
# Also need to copy any other necessary files if needed, but outputs contains model.joblib

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
