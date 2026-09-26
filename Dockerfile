# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl and certificates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Install uv and remove the installer
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy dependency files
COPY pyproject.toml uv.lock /code/

# Install Python dependencies
RUN uv sync --frozen

# Download the spaCy large English model
RUN uv run python -m spacy download en_core_web_lg

# Copy the application code
COPY ./app /code/app

# Run the FastAPI application
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]