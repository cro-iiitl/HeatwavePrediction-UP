"""
One-time utility to compute UP district centroids from GADM boundaries.
NOT part of the runtime pipeline — src/data/ never imports this.
Output feeds configs/data.yaml generation (scripts/generate_data_yaml.py).
"""

import geopandas as gpd
import pyogrio

GADM_PATH = "scripts/gis_data/gadm41_IND.gpkg"
OUTPUT_CSV = "scripts/district_centroids_raw.csv"

layers = pyogrio.list_layers(GADM_PATH)
print("Available layers:")
print(layers)

layer_names = layers[:, 0]
DISTRICT_LAYER = [l for l in layer_names if "2" in l][0]
print(f"Using layer: {DISTRICT_LAYER}")

gdf = gpd.read_file(GADM_PATH, layer=DISTRICT_LAYER)

print("Columns available:", gdf.columns.tolist())
print("States found:", gdf["NAME_1"].unique() if "NAME_1" in gdf.columns else "NAME_1 not found — check columns above")

up = gdf[gdf["NAME_1"] == "Uttar Pradesh"].copy()
print(f"\nFound {len(up)} districts in source data — expected 75")

if len(up) != 75:
    print("WARNING: count mismatch. Do not proceed to YAML generation until reconciled.")

up_projected = up.to_crs(epsg=7755)
centroids_projected = up_projected.geometry.centroid

centroids_wgs84 = gpd.GeoSeries(centroids_projected, crs=7755).to_crs(epsg=4326)

up["lon"] = centroids_wgs84.x
up["lat"] = centroids_wgs84.y

result = up[["NAME_2", "lat", "lon"]].rename(columns={"NAME_2": "district_name"})
result = result.sort_values("district_name").reset_index(drop=True)

result.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(result)} centroids to {OUTPUT_CSV}")
print(result.head(10))