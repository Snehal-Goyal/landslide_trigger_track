# Landslide Inventory Processing and Classification

This project processes landslide inventory data using topological data analysis (TDA) and machine learning. It extracts features from 3D polygon representations of ground failures and classifies them based on their triggering mechanism (e.g., earthquake or rainfall).

## Features
- Reads shapefiles of landslide polygons.
- Generates 3D polygon representations using Digital Elevation Models (DEMs).
- Extracts topological and machine learning features in batches.
- (Optional) Classifies landslides into earthquake-triggered or rainfall-triggered categories.

## Requirements
### Dependencies
Install the following Python libraries:
- `numpy`
- `geopandas`
- `scikit-learn`
- `matplotlib`
- `xgboost`

Use the following command to install dependencies (if not already installed):

```bash
pip install -r requirements.txt
```

### File Structure
Ensure your project directory contains the following files and folders:

```
project/
|-- main.py
|-- topological_features_based_model.py
|-- data/
|   |-- ground_failure_polygons.shp
|   |-- japan/
|       |-- Earthquake_hokkaido_polygons.shp
|       |-- Earthquake_iwata_polygons.shp
|       |-- Earthquake_niigata_polygons.shp
|       |-- Rainfall_kumamoto_polygons.shp
|       |-- Rainfall_fukuoka_polygons.shp
|       |-- Rainfall_saka_polygons.shp
|-- dem/
|   |-- greece.tif
```

## How to Run

1. **Prepare the Environment**
   - Ensure that shapefiles for Greece and Japanese inventories are present in the `data/` folder.
   - Confirm the DEM files are located in the `dem/` folder. If not, the script will attempt to download them automatically.

2. **Run the Script**
   Execute the `main.py` script using the following command:

   ```bash
   python main.py
   ```

3. **Expected Output**
   - The script processes the first 10 polygons from Greece for testing.
   - A plot of the first polygon is saved in the `plots/` directory.
   - Extracted features are saved in:
     - `features_greece.xlsx`
     - `features_greece.csv`

4. **Scaling Up**
   To process the entire dataset, remove or modify the line restricting Greece polygons to the first 10 entries:
   ```python
   greece_polygons = greece_polygons.iloc[:10]
   ```

   Adjust `batch_size` in the script for optimal performance with larger datasets.

## Project Workflow

1. **Shapefile Reading**
   - Reads input shapefiles and filters non-liquefaction polygons.
   - Subsets Greece polygons based on the `epicentral` field.

2. **3D Polygon Generation**
   - Generates 3D polygon representations using DEMs.
   - Stores 3D point cloud data in `.npz` format for future use.

3. **Feature Extraction**
   - Extracts ML-compatible features from the 3D polygons in batches.
   - Saves features in `.xlsx` and `.csv` formats.

4. **(Optional) Classification**
   - Combines Japanese earthquake and rainfall inventory features.
   - Uses `XGBoost` to classify Greece polygons as earthquake or rainfall triggered.

## Customization
- Modify `batch_size` to balance performance and resource usage.
- Update file paths for different shapefiles and DEM locations.
- Extend or customize feature extraction and classification steps in `topological_features_based_model.py`.

## Known Issues
1. **Performance**: Processing large datasets may take significant time. Use batching to improve efficiency.
2. **Non-Numeric Values**: Ensure all features are numeric. The script will raise an error if non-numeric values are encountered.
3. **DEM Files**: Missing DEM files will trigger an automatic download. Ensure sufficient disk space and a stable internet connection.

## Future Improvements
- Implement parallel processing for 3D polygon generation and feature extraction.
- Add logging to improve debugging and traceability.
- Include more detailed visualization for processed polygons and classification results.

## License
This project is licensed under the MIT License. See the LICENSE file for details.



For questions or issues, please contact [snehal@flaxandteal@co.uk].

