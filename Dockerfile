FROM kennethreitz/pipenv

COPY . /app

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=America/Los_Angeles
RUN apt install dialog -y
RUN apt install python3.6-tk -y

RUN pipenv update

CMD pipenv run python main.py
