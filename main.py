import os
import numpy as np
import geopandas as gpd
from topological_features_based_model import (
    read_shapefiles,
    plot_polygon,
    download_dem,
    make_3d_polygons,
    get_ml_features_with_files,
    classify_inventory_tda_with_xgboost,
)


def main():
    print("Starting main...")

    # File paths
    shapefile_path = "/home/snehal/Downloads/final/data/ground_failure_polygons.shp"
    dem_location = "./dem"
    inventory_name = "greece.tif"

    # Updated storage file for point cloud (using NumPy format)
    npz_pointcloud_file = "pointcloud_Greece.npz"

    # Output feature files
    features_excel = "features_greece.xlsx"
    features_csv = "features_greece.csv"

    # 1. Read the shapefile
    print("Reading shapefile...")
    ground_failures = read_shapefiles(shapefile_path)

    # Filter out liquefaction
    ground_failures = ground_failures[ground_failures['type'] != 'Liquefaction'].dropna(subset=['geometry'])

    # Select only Greece polygons; you'll adjust this condition to match your data
    greece_polygons = ground_failures[ground_failures['epicentral'] == 'Greece'].reset_index(drop=True)

    # Subset for testing (e.g., first 10 polygons)
    greece_polygons = greece_polygons[['geometry']].iloc[:10]
    print(f"Number of polygons in subset: {len(greece_polygons)}")

    # 2. Plot a sample polygon (optional)
    print("Plotting the first polygon (index=0)...")
    plot_polygon(poly_data=greece_polygons, polygon_index=0)

    # 3. Check if DEM exists; if not, download
    dem_file = os.path.join(dem_location, inventory_name)
    if not os.path.exists(dem_file):
        print(f"DEM file not found. Downloading to: {dem_file}")
        download_dem(greece_polygons, dem_location, inventory_name)
    else:
        print(f"DEM file already exists: {dem_file}")

    # 4. Generate or load 3D polygons (point cloud)
    if os.path.exists(npz_pointcloud_file):
        print(f"Loading 3D polygons from {npz_pointcloud_file}...")
        npz_data = np.load(npz_pointcloud_file)
        pointcloud_Greece = [npz_data[f"arr_{i}"] for i in range(len(npz_data.files))]
    else:
        print("3D polygons not found. Generating 3D polygons...")
        pointcloud_Greece = make_3d_polygons(
            poly_data=greece_polygons,
            dem_location=dem_location,
            inventory_name=inventory_name,
            use_existing_dem=True
        )
        print("3D polygons generated successfully.")

        np.savez_compressed(npz_pointcloud_file, *pointcloud_Greece)
        print(f"3D point cloud data saved to {npz_pointcloud_file} (compressed NPZ).")

    # 5. Extract TDA/ML features
    print("Extracting ML features from point cloud...")
    features_greece = get_ml_features_with_files(
        pointcloud_Greece,
        output_excel=features_excel,
        output_csv=features_csv
    )
    print(f"ML features saved to {features_excel} and {features_csv}")
    print("Done.")

    earth_hokkaido_shapefile = read_shapefiles("/home/snehal/Downloads/final/data/japan/Earthquake_hokkaido_polygons.shp")
    earth_iwata_shapefile = read_shapefiles("/home/snehal/Downloads/final/data/japan/Earthquake_iwata_polygons.shp")
    earth_niigata_shapefile = read_shapefiles("/home/snehal/Downloads/final/data/japan/Earthquake_niigata_polygons.shp")
    rain_kumamoto_shapefile = read_shapefiles("/home/snehal/Downloads/final/data/japan/Rainfall_kumamoto_polygons.shp")
    rain_fukuoka_shapefile = read_shapefiles("/home/snehal/Downloads/final/data/japan/Rainfall_fukuoka_polygons.shp")
    rain_saka_shapefile = read_shapefiles("/home/snehal/Downloads/final/data/japan/Rainfall_saka_polygons.shp")

    japan_inventory_name_list = ['hokkaido.tif', 'iwata.tif', 'niigata.tif', 'kumamoto.tif', 'fukuoka.tif', 'saka.tif']

    dem_file = download_dem(earth_hokkaido_shapefile, dem_location, japan_inventory_name_list[0])
    dem_file = download_dem(earth_iwata_shapefile, dem_location, japan_inventory_name_list[1])
    dem_file = download_dem(earth_niigata_shapefile, dem_location, japan_inventory_name_list[2])
    dem_file = download_dem(rain_kumamoto_shapefile, dem_location, japan_inventory_name_list[3])
    dem_file = download_dem(rain_fukuoka_shapefile, dem_location, japan_inventory_name_list[4])
    dem_file = download_dem(rain_saka_shapefile, dem_location, japan_inventory_name_list[5])

    pointcloud_earth_hokkaido = make_3d_polygons(earth_hokkaido_shapefile, dem_location, japan_inventory_name_list[0], 1)
    pointcloud_earth_iwata = make_3d_polygons(earth_iwata_shapefile, dem_location, japan_inventory_name_list[1], 1)
    pointcloud_earth_niigata = make_3d_polygons(earth_niigata_shapefile, dem_location, japan_inventory_name_list[2], 1)
    pointcloud_rain_kumamoto = make_3d_polygons(rain_kumamoto_shapefile, dem_location, japan_inventory_name_list[3], 1)
    pointcloud_rain_fukuoka = make_3d_polygons(rain_fukuoka_shapefile, dem_location, japan_inventory_name_list[4], 1)
    pointcloud_rain_saka = make_3d_polygons(rain_saka_shapefile, dem_location, japan_inventory_name_list[5], 1)


    features_earth_iwata=get_ml_features_with_files(pointcloud_earth_iwata)
    features_earth_hokkaido=get_ml_features_with_files(pointcloud_earth_hokkaido)
    features_earth_niigata=get_ml_features_with_files(pointcloud_earth_niigata)
    features_rain_kumamoto=get_ml_features_with_files(pointcloud_rain_kumamoto)
    features_rain_fukuoka=get_ml_features_with_files(pointcloud_rain_fukuoka)
    features_rain_saka=get_ml_features_with_files(pointcloud_rain_saka)


    Total_earthquake_inventory_features_Japan = np.vstack((features_earth_hokkaido, features_earth_iwata, features_earth_niigata))
    Total_rainfall_inventory_features_Japan = np.vstack((features_rain_kumamoto, features_rain_fukuoka, features_rain_saka))
    test_inventory_features = features_greece

    predict_probability_greece = classify_inventory_tda_with_xgboost(
        Total_earthquake_inventory_features_Japan,
        Total_rainfall_inventory_features_Japan,
        test_inventory_features
    )


if __name__ == "__main__":
    main()
