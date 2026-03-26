# Prithvi EO 2.0 Hazard Mapping - QGIS Plugin

This project provides a QGIS Plugin that leverages the **IBM-NASA Prithvi EO 2.0 Geospatial Foundation Model (GFM)** for automated hazard detection (specifically flood mapping and burn scars detection) using satellite data.

The plugin uses a **Client-Server (Hybrid)** architecture to keep it lightweight in the QGIS environment, while the heavier AI inference is performed in a separate local Python process.

## Architecture Overview
- **Frontend:** A QGIS Plugin written in Python/PyQt that handles data selection and visualization.
- **Backend:** A FastAPI server that runs the Prithvi model, downloaded from Hugging Face Hub, using ```terratorch```

## Quick Start Guide

**1. Download the Project**
First, get the code onto your local machine.

### Option A: Using Git (Recommended)

```bash
git clone https://github.com/YourUsername/hazard-detection-project.git
cd hazard-detection-project
```
### Option B: Manual Download

1. Download the repository as a ZIP file from GitHub.
2. Extract it to a folder (e.g., C:\Users\USER\Documents\Hazard_Detection).
3. Open your terminal/command prompt and ```cd``` into that extracted folder.

**2. Setup the AI Backend** 
### A. Create and Activate Environment
```bash
cd backend
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate
```

### B. Install Dependencies
```bash
pip install -r requirements.txt
```

### C. Start the Server
```bash
python -m uvicorn main:app --reload
```
*Note: On the first run, the backend will automatically download the ~1.2GB Prithvi model weights from Hugging Face. This may take a few minutes depending on your internet speed.*

**3. Install the QGIS Plugin**
...
## Usage Instructions

## Requirements

## Acknowledgements

## Licence
