FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "demo_play_with_model:app", "-c", "./gunicorn.conf.py"]
#CMD [ "python", "./demo_play_with_model.py" ]
#ENV FLASK_APP=demo_play_with_model
#CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5556"]