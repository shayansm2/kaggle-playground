FROM python:3.8.12-slim
LABEL authors="shayan"
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["../py-scripts/gateway.py", "../py-scripts/protobuf.py", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"]



