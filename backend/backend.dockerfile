FROM python:3.12-slim

WORKDIR /app

COPY src/requirements.txt src/

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8080"]
