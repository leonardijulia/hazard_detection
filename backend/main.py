import os 
import yaml 
import torch
import numpy as np
import rasterio as rio
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import  FileResponse
from contextlib import asynccontextmanager
from terratorch.tasks import SemanticSegmentationTask
from terratorch.tasks.tiled_inference import tiled_inference

def load_model():
    REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11" # for hf-hub in the next round of enhancements
    print("Initializing Prithvi EO 2.0...")
    
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="Prithvi-EO-V2-300M-TL-Sen1Floods11.pt")
    configs_local_path = hf_hub_download(repo_id=REPO_ID, filename="config.yaml")
    
    with open(configs_local_path, "r") as f:
        config = yaml.safe_load(f)
    
    model = SemanticSegmentationTask.load_from_checkpoint(
        weights_path,
        **config['model']['init_args']['model_args'])
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model
    
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    model_cache["model"] = load_model()

    yield
    model_cache.clear()

app = FastAPI(lifespan=lifespan)
    
def preprocess_and_predict(input_path, output_path, sensor_type="S2_L1C"):
    model = model_cache["model"]
    #device = model_cache["device"]
    
    with rio.open(input_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile
        
    if sensor_type in ["S2_L1C", "S2_L2A"]:
        data = data * 0.0001
    if sensor_type == "L8_L2":
        data = (data * 0.0000275) - 0.2
    
    means = np.array([0.1412956, 0.13795798, 0.12353792, 0.30902815, 0.2044958, 0.11912015])
    stds = np.array([0.07406382, 0.07370365, 0.08692279, 0.11798815, 0.09772074, 0.07659938])
    
    data = (data - means.reshape(6, 1, 1)) / stds.reshape(6, 1, 1)
    input_tensor = torch.from_numpy(data).unsqueeze(0).float() # permute checking if the dat ais in the H, W, C format first!
    
    def model_forward(x, **kwargs):
        return model(x, **kwargs).output

    with torch.no_grad():
        pred = tiled_inference(model_forward, input_tensor, crop=224, stride=200)
        mask = pred.squeeze().argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    profile.update(dtype=rio.uint8, count=1, nodata=None)
    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(mask, 1)
        
@app.post("/predict")
async def predict_flood(file: UploadFile = File(...), sensor: str = "S2_L1C"):
    input_path = f"temp_in_{file.filename}"
    output_path = f"mask_out_{file.filename}"
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        preprocess_and_predict(input_path, output_path, sensor_type=sensor)
        # return {"status": "success", "mask_path": os.path.abspath(output_path)}
        return FileResponse(
            path=output_path,
            media_type='image/tiff',
            filename=f"mask_{file.filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))