FROM python:3.9-slim

WORKDIR /app/src

COPY . /app

COPY ./data /app/data

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
