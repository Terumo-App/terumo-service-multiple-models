from fastapi import FastAPI, File, UploadFile,HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import json
from base_model import Model
from schema.schemas import HealthCheckResult
from PIL import Image
import io
import base64
import logging
import uvicorn
from pydantic import BaseModel



class ImagePayload(BaseModel):
    image: dict


logging.getLogger().setLevel(logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = [
    Model('HIPER'),
    Model('MEMBR'),
    Model('NORM'),
    Model('SCLER'),
    Model('PODOC'),
    Model('CRESC'),
    ]
MODELS_NAME = [
    'Hiperceluraridade',
    'Membranosa',
    'Normal',
    'Screrosis',
    'Podocitopatia',
    'Crescente',
    ]

def convert_base64_to_image(image: str)->Image.Image:
    """ this function decodes base64 image data to binary and afeter that convert it
    for Pillow image object.

    Args:
        image (str): image encoded in base64

    Returns:
        PIL.PngImagePlugin.PngImageFile
    
    """
    # Decode base64 image data
    image_data = base64.b64decode(image)
    # Convert bytes to image object
    return Image.open(io.BytesIO(image_data))

# Default route
@app.post("/predict")
async def process_image(body: ImagePayload):
    if  'data' not in body.image.keys():
        raise HTTPException(status_code=400, detail="No image data provided")  
    
    image = convert_base64_to_image(body.image['data'])
    
    predictions = []
    for model, name in zip(MODELS, MODELS_NAME):
        image_processed = model.process(image) 
        prediction = model.predict(image_processed) 
        predictions.append({ 'model_name': name, 'prediction':prediction,})

    # print(predictions)

    return predictions


@app.get("/health", response_model=HealthCheckResult)
def health_check() -> HealthCheckResult:
    return HealthCheckResult(success=True)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
