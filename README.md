# Artisanal Small-Scale Gold Mining Detection in the Amazon (Thesis Project)
## Master Thesis Information Studies 2024 - 2025

This repository contains the code and analysis for my Master's thesis focused on detecting illegal artisanal gold mining in the Bolívar region of the Amazon Forest using satellite imagery and contextual geographic data. The goal is to train a model that can identify mining activity from Sentinel-2 patches, enriched with information from OpenStreetMap (OSM).

---

<!-- ## What This Project Does

- Generates training data by extracting 48×48 Sentinel-2 image patches from known GPS-labeled sites (positive = mining, negative = non-mining).
- Combines visual data with contextual information from OSM (e.g., roads, rivers, land use).
- Trains a classifier to distinguish between mining and non-mining regions using remote sensing and geographic context.

--- -->

## Built On Top Of

This work builds upon the [Earthrise Media Mining Detector](https://github.com/earthrise-media/mining-detector), an open-source project designed to automate the detection of mining scars in Amazonian satellite imagery. Their codebase laid the groundwork for patch generation, Sentinel-2 preprocessing, and basic model structure and training.

For my thesis, I adapted parts of their pipeline to include OSM features.

---

<!-- ## Why This Matters

Artisanal mining leaves visible scars in the rainforest—muddy flats, wastewater pools, deforestation—that can be picked up in satellite imagery. However, these patterns are subtle and require careful analysis to distinguish from natural variation. By combining raw imagery with human-generated map data, we hope to improve the model’s ability to generalize and support environmental monitoring efforts.

--- -->

<!-- ## Project Structure
data/
├── boundaries/               # Area-of-interest GeoJSONs
├── sampling_locations/       # GPS-labeled mine/non-mine locations
├── contextual/               # OSM data extracted for the region
├── training_data/            # Extracted 48×48 image patches and labels

gee/
├── get_training_data.py      # Patch creation logic (adapted from Earthrise)
├── gee.py                    # Google Earth Engine data extraction

notebooks/
├── Exploratory Data Analysis.ipynb  # Full EDA of the dataset

--- -->


## License

Parts of this project build on the original Earthrise code, which is released under the [MIT License](https://github.com/earthrise-media/mining-detector/blob/main/LICENSE). 

---

## Acknowledgements

Thanks to the Earthrise team for making their mining detector pipeline open-source.
