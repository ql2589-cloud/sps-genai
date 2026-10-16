
import io
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.cnn_model import CNNModel


app = FastAPI()


# 1. Bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


# 2. Embedding model
embedding_model = EmbeddingModel()


class EmbeddingRequest(BaseModel):
    word: str


# 3. CNN image classifier
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "cnn_model.pth"

cnn_model = CNNModel()

cnn_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
)

cnn_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# 4. API endpoints
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.calculate_embedding(
        request.word
    )
    return {
        "word": request.word,
        "embedding": embedding
    }


@app.post("/classify-image")
def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = file.file.read()
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_class = CIFAR10_CLASSES[
        predicted_index.item()
    ]

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4)
    }
