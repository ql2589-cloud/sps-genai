# SPS Generative AI - Assignment 1

This project implements a FastAPI application with bigram text generation and spaCy word embeddings.

## Features

- Bigram text generation
- Word embeddings using spaCy `en_core_web_lg`
- FastAPI endpoints for model inference
- Docker deployment

## Run with Docker

Build the Docker image:

```bash
docker build -t sps-genai .
```

Run the container:

```bash
docker run -p 8000:80 sps-genai
```

Open the FastAPI documentation in a browser:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### POST /generate

Example request:

```json
{
  "start_word": "the",
  "length": 10
}
```

### POST /embedding

Example request:

```json
{
  "word": "apple"
}
```

The `/embedding` endpoint returns the input word and its embedding vector generated using spaCy `en_core_web_lg`.