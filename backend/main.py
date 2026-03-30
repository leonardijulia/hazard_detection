import os 
import yaml 
import torch
import numpy as np
import rasterio as rio
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import  FileResponse
from contextlib import asynccontextmanager
from terratorch.tasks import SemanticSegmentationTask
from terratorch.tasks.tiled_inference import tiled_inference
    
model_cache = {}

def load_model(hazard:str):
    REPO_ID_FLOODS = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"  
    REPO_ID_BURN = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
    
    weights_path_floods = hf_hub_download(repo_id=REPO_ID_FLOODS, filename="Prithvi-EO-V2-300M-TL-Sen1Floods11.pt")
    configs_local_path_floods = hf_hub_download(repo_id=REPO_ID_FLOODS, filename="config.yaml")
    
    weights_path_burn = hf_hub_download(repo_id=REPO_ID_BURN, filename="Prithvi_EO_V2_300M_BurnScars.pt")
    configs_local_path_burn = hf_hub_download(repo_id=REPO_ID_BURN, filename="burn_scars_config.yaml")
    
    if hazard == "floods":
        configs_local_path = configs_local_path_floods
        weights_path = weights_path_floods
    elif hazard == "burn_scars":
        configs_local_path = configs_local_path_burn
        weights_path = weights_path_burn
    else:
        raise ValueError(f"Unsupported hazard type: '{hazard}'. Supported types are 'floods' or 'burn_scars'.")
    
    with open(configs_local_path, "r") as f:
        config = yaml.safe_load(f)
    
    model = SemanticSegmentationTask.load_from_checkpoint(
        weights_path,
        **config['model']['init_args']['model_args'])
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load BOTH on startup
    print("Pre-loading Prithvi Models (this may take a minute)...")
    try:
        model_cache["floods"] = load_model("floods")
        model_cache["burn_scars"] = load_model("burn_scars")
    except Exception as e:
        print(f"Error loading models during startup: {e}")
    
    yield
    model_cache.clear()

app = FastAPI(lifespan=lifespan)

def get_model(hazard: str):
    if hazard not in model_cache:
        raise HTTPException(status_code=500, detail=f"Model {hazard} not initialized.")
    return model_cache[hazard]
    
def preprocess_and_predict(input_path, output_path, model, hazard="floods", sensor_type="S2_L1C"):
    
    with rio.open(input_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile
        
    data_max = np.max(data)
    
    if sensor_type in ["S2_L1C", "S2_L2A"]:
        if data_max > 2:
            data = data * 0.0001
    elif sensor_type == "L8_L2":
        if data_max > 2:
            data = (data * 0.0000275) - 0.2
            
    if hazard == "floods":
        means = np.array([0.1412956, 0.13795798, 0.12353792, 0.30902815, 0.2044958, 0.11912015])
        stds = np.array([0.07406382, 0.07370365, 0.08692279, 0.11798815, 0.09772074, 0.07659938])
    elif hazard == "burn_scars":
        means = np.array([0.0333497, 0.05701186, 0.05889748, 0.23232451, 0.1972855, 0.11944914])
        stds = np.array([0.02269135, 0.02680756, 0.04004109, 0.07791732, 0.0870874, 0.07241979])
    
    data = (data - means.reshape(6, 1, 1)) / stds.reshape(6, 1, 1)
        
    input_tensor = torch.from_numpy(data).unsqueeze(0).float() # permute checking if the dat ais in the H, W, C format first!
    
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    def model_forward(x, **kwargs):
        return model(x, **kwargs).output

    with torch.no_grad():
        pred = tiled_inference(model_forward, input_tensor, crop=224, stride=200)
        mask = pred.squeeze().argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    profile.update(dtype=rio.uint8, count=1, nodata=None)
    with rio.open(output_path, 'w', **profile) as dst:
        dst.write(mask, 1)
 
@app.get("/health")
async def health_check():
    """
    Simple endpoint for the QGIS plugin to verify 
    the backend is online without running AI inference.
    """
    return {"status": "ok", "model": "Prithvi EO 2.0"}
      
@app.post("/predict")
async def predict_flood(file: UploadFile = File(...), sensor: str = Form("S2_L1C"), hazard: str = Form("floods")):
    input_path = f"temp_in_{file.filename}"
    output_path = f"mask_out_{file.filename}"
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        model = get_model(hazard)
        preprocess_and_predict(input_path, output_path, model, hazard=hazard, sensor_type=sensor)
        # return {"status": "success", "mask_path": os.path.abspath(output_path)}
        return FileResponse(
            path=output_path,
            media_type='image/tiff',
            filename=f"mask_{file.filename}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))