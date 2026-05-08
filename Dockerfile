FROM python:3.11

RUN apt-get update && \
    apt-get install -y tesseract-ocr

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "bot.py"]
