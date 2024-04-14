# Use Python 3.12 as the builder image
FROM python:3.12 as builder

ENV POETRY_VERSION=1.2.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /build
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy over project files in 'src' directory
COPY src /build/src

# Final stage with the runtime environment
FROM python:3.12-slim
COPY --from=builder /build /app
WORKDIR /app
CMD ["python", "src/main.py"]
