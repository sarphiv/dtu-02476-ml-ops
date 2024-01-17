FROM python:3.10-slim

WORKDIR /app

ENV PORT=80
EXPOSE ${PORT}
ENV INFERENCE_API_URL=localhost:8080

COPY requirements_website.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/website/main.py .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
