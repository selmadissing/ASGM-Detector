# GEE Patch Extraction

This folder contains code to extract Sentinel-2 image patches using Google Earth Engine (GEE) for ASGM (artisanal and small-scale gold mining) detection. Patches are generated based on sampling locations from GeoJSON files and saved for training deep learning models.

---

## Overview

The patch extraction workflow:

1. Loads sampling tiles (e.g., from `positive.geojson`, `negative.geojson`)
2. Uses the GEE API to retrieve Sentinel-2 imagery for each tile
3. Applies cloud masking (for S2) and constructs a median composite
4. Clips and exports image patches as NumPy arrays
5. Saves patches and tile metadata to disk

---

## GEE Authentication Setup (Required Before Running Scripts)

Before running any script using Earth Engine (e.g. `get_training_data.ipynb`), you **must authenticate manually** using the `earthengine` CLI.

This step is required **once per environment** and ensures the Python API has access to your GEE account and project.

### Step 1: Clear previous authentication (recommended)

```bash
gcloud auth application-default revoke
gcloud auth revoke
rm -rf ~/.config/gcloud
rm -rf ~/.config/earthengine
```

### Step 2: Authenticate Earth Engine using notebook mode

```bash
earthengine authenticate --auth_mode=notebook --force
```
This will open a browser for login and generate the credentials file here: `~/.config/earthengine/credentials`


### Step 3: Update Your Project ID

The script gee.py includes the line: `ee.Initialize(project='mining-thesis-selma')`

You must replace 'mining-thesis-selma' with your own Google Cloud project ID that has Earth Engine enabled.

If you do not have one:
1.	Create a new project at console.cloud.google.com
2.	Enable the Earth Engine API for your project
3.	Ensure your account has permission to access it

Once this setup is complete, you can run the patch extraction