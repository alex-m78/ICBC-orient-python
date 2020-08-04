FROM python

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY / /app

EXPOSE 20880

CMD ["python","main.py"]