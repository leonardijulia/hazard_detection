# Prithvi EO 2.0 Hazard Mapping - QGIS Plugin

This project provides a QGIS Plugin that leverages the **IBM-NASA Prithvi EO 2.0 Geospatial Foundation Model (GFM)** for automated hazard detection (specifically flood mapping and burn scars detection) using satellite data.

The plugin uses a **Client-Server (Hybrid)** architecture to keep it lightweight in the QGIS environment, while the heavier AI inference is performed in a separate local Python process.

## Architecture Overview
- **Frontend:** A QGIS Plugin written in Python/PyQt that handles data selection and visualization.
- **Backend:** A FastAPI server that runs the Prithvi model, downloaded from Hugging Face Hub, using ```terratorch```

## Quick Start Guide

### **1. Download the Project**
First, get the code onto your local machine.

##### Option A: Using Git (Recommended)

```bash
git clone https://github.com/leonardijulia/hazard_detection.git
cd GFM_plugin
```
##### Option B: Manual Download

1. Download the repository as a ZIP file from GitHub.
2. Extract it to a folder (e.g., C:\Users\USER\Documents\Hazard_Detection).
3. Open your terminal/command prompt and ```cd``` into that extracted folder.

### **2. Setup the AI Backend** 
#### A. Create and Activate Environment
```bash
cd backend
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate
```

#### B. Install Dependencies
```bash
pip install -r requirements.txt
```

#### C. Set up Hugging Face authentication (optional but recommended)
For a quicker download of the model weights when starting up the server it is recommended to generate a [Hugging Face](https://huggingface.co) access token by following the instructions [here](https://huggingface.co/docs/hub/en/security-tokens).

Run the following command in your terminal to log in to Hugging Face:
```bash
hf auth login
```
Provide your access token when prompted.

#### D. Start the Server
You can start the server in two ways:

**Option 1: Quick Start (Windows)**
Simply double-click the start_backend.bat file in the backend folder. This will automatically use the virtual environment to launch the server.

**Option 2: Manual Start (CLI)**
- *Windows*
```bash
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

- *Linux/macOS*
```bash
./.venv/bin/python -m uvicorn main:app --reload
```

*Note: On the first run, the backend will automatically download the Prithvi models weights from Hugging Face. This may take a few minutes depending on your internet speed. The server is ready when you see ```Uvicorn running on http://127.0.0.1:8000```*

### **3. Install the QGIS Plugin**
1. Find your QGIS plugins directory:
- Windows: ```%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins```
- macOS: ```~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins```
- Linux: ```~/.local/share/QGIS/QGIS3/profiles/default/python/plugins```

2. Copy or symlink the ```qgis_plugin``` folder to the plugins directory and rename it to ```hazard_detection```

3. Restart QGIS

4. Enable the plugin: Go to Plugins > Manage and Install Plugins, find "Prithvi Hazard Detector", and enable it.

## Usage Instructions
1. **Load Data:** Add a Sentinel-2 or Landsat image to your QGIS project.

2. **Open Plugin:** Click the Prithvi Hazard Detection plugin icon in the toolbar.

3. **Map Bands:** Select the corresponding bands for Blue, Green, Red, NIR, SWIR1, and SWIR2. The plugin will automatically reorder these into the required HLS format.

4. **Set URL:** Ensure the Server URL is set to http://127.0.0.1:8000.

5. **Run:** Click OK. The plugin will create a temporary stack, ship it to the backend, and load the resulting Flood Mask back into your map.

## Requirements
- Python 3.10+
- QGIS 3.22+
- ```torch```
- ```terratorch```
- ```fastapi``` & ```uvicorn```
- ```huggingface_hub```
- ```rasterio```

## Acknowledgements
- **IBM & NASA:** For the Prithvi EO 2.0 Foundation Model.

Szwarcman, D., Roy, S., Fraccaro, P., et al. (2024). *Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications.* arXiv preprint arXiv:2412.02732. Available at: https://arxiv.org/abs/2412.02732

## Licence
This project is licensed under the **GNU General Public License v3.0.**
