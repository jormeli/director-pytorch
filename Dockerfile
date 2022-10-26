FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION_SHA=88ba18d5d886195e48098e8974157d7a1089b1d1

RUN apt-get update
RUN apt-get install -y git

RUN pip install git+https://github.com/python-poetry/poetry.git@$POETRY_VERSION_SHA

WORKDIR /app
COPY poetry.lock pyproject.toml /app

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

COPY . .
