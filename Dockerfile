FROM kennethreitz/pipenv

COPY . /app

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=America/Los_Angeles
RUN apt install dialog python3.6-tk python-opengl xvfb -y

RUN pipenv update

CMD pipenv run xvfb-run -s "-screen 0 1400x900x24" python main.py