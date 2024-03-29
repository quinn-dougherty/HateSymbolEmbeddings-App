FROM python:3.7-slim-buster
WORKDIR /code
ENV FLASK_APP hatesymbolembeddings:APP
ENV FLASK_RUN_HOST 0.0.0.0
RUN apt-get update -qqy && apt-get install -qqy libopenblas-dev gfortran
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE $PORT

COPY . . 

# RUN mkdir tmp
#CMD ["flask", "run"]
#CMD ["python", "hatesymbolembeddings/app.py"]
# CMD ["gunicorn", "hatesymbolembeddings:APP", "-b", "0.0.0.0:33507"]
CMD gunicorn hatesymbolembeddings:APP --bind 0.0.0.0:$PORT
