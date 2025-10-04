FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "src.infer:app", "--host", "0.0.0.0", "--port", "8000"]
