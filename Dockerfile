FROM kennethreitz/pipenv

COPY . /app

RUN apt-get install python3.6-tk -y

RUN pipenv update

CMD pipenv run python main.py