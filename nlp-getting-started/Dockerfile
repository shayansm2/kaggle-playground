FROM python:latest
LABEL authors="shayan"
ENV PYTHONUNBUFFERED=TRUE
RUN pip --no-cache-dir install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system
RUN rm -rf /root/.cache
RUN python -c "import nltk; nltk.download('stopwords')"
COPY ["*.py", "*.bin", "./"]
EXPOSE 1234
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:1234", "web_server:app"]