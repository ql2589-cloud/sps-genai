
# Use an official Python runtime
FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /code

# Copy project dependency files
COPY pyproject.toml uv.lock /code/

# Install Python 3.13 and project dependencies
RUN uv python install 3.13
RUN uv sync --frozen --python 3.13

# Download spaCy English model
RUN uv run python -m spacy download en_core_web_lg

# Copy application code
COPY ./app /code/app

# Copy trained CNN weights
COPY ./cnn_model.pth /code/cnn_model.pth

# Expose the API port
EXPOSE 80

# Start FastAPI
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
