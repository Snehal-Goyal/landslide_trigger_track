# **Landslide Inventory Processing and Classification**

This project focuses on processing landslide inventory data using **Topological Data Analysis (TDA)** and machine learning. It extracts features from 3D polygon representations of ground failures and classifies them based on their triggering mechanisms (e.g., earthquake or rainfall).

---

## **Features**

- **Shapefile Processing**: Reads shapefiles containing landslide polygons.
- **3D Polygon Generation**: Creates 3D polygon representations using Digital Elevation Models (DEMs).
- **Topological Feature Extraction**: Extracts features using TDA.
- **Landslide Classification**: Classifies landslides as earthquake-triggered or rainfall-triggered using **XGBoost**.

---

## **Requirements**

### **Dependencies**
Install the following Python libraries:
- `numpy`
- `geopandas`
- `scikit-learn`
- `matplotlib`
- `xgboost`

To install all dependencies, use the command:
```bash
pip install -r requirements.txt
```

---

## **File Structure**

Ensure your project directory contains the following structure:

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
|   |-- greece.tif (or your testing inventory)
```

---

## **How to Run**

### **1. Prepare the Environment**
- Ensure the shapefiles for Greece (or any other testing inventory) and Japanese inventories (or other training inventories) are present in the `data/` folder.
- Confirm that DEM files are available in the `dem/` folder. If not, the script will automatically download them.

### **2. Execute the Script**
Run the `main.py` script with the following command:
```bash
python main.py
```

### **3. Output**
- A plot of the first polygon will be saved in the `plots/` directory for visualization.
- Extracted features will be saved as:
  - `features_greece.xlsx`
  - `features_greece.csv`

---

## **Project Workflow**

### **1. Shapefile Reading**
- Reads input shapefiles and filters out non-liquefaction polygons.
- Subsets Greece polygons based on the **epicentral field**.

### **2. 3D Polygon Generation**
- Generates 3D polygon representations using DEMs.
- Saves 3D point cloud data in `.npz` format for reuse.

### **3. Feature Extraction**
- Extracts machine learning-compatible features from the 3D polygons using **Topological Data Analysis (TDA)**.
- Saves features in `.xlsx` and `.csv` formats.

### **4. Classification**
- Combines Japanese earthquake and rainfall inventory features.
- Uses **XGBoost** to classify Greece polygons as earthquake-triggered or rainfall-triggered.

---
Refer to the report folder provided above .


## **Known Issues**

1. **Performance**:
   - Processing large datasets can be time-consuming. Consider batching to improve efficiency.
2. **Non-Numeric Values**:
   - Ensure all features are numeric. Non-numeric values will raise an error.
3. **DEM Files**:
   - Missing DEM files trigger an automatic download. Ensure you have sufficient disk space and a stable internet connection.

---

## **Contact**
For any questions or issues, please contact:  
**Snehal Goyal**  
Email: [snehal@flaxandteal.co.uk](mailto:snehal@flaxandteal.co.uk)

