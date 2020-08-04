FROM python

WORKDIR /app

COPY python/requirements.txt /app/

RUN pip install -r requirements.txt

COPY python/ /app

EXPOSE 20880

CMD ["python","main.py"]
