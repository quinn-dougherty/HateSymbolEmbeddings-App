FROM python:3.7-slim-buster
WORKDIR /code
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0
RUN apt-get update -qqy && apt-get install -qqy libopenblas-dev gfortran
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir tmp
CMD ["flask", "run"]