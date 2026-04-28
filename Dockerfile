FROM python:3.11-slim

WORKDIR /app

COPY main.py /app/
COPY infer.py /app/
COPY models /app/models

RUN pip install fastapi uvicorn onnx onnxruntime numpy

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]