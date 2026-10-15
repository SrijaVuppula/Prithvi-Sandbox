"""
Export lat/long center coordinates for every chip in training_chips/ as a KML
file, viewable in Google Earth / Google Maps.

By default all *_merged.tif chips in the folder are exported (the full dataset).
Pass --study-only to restrict to the 500-chip verified study set, read from the
'chip' column of results_600M.csv.

No acquisition date is included -- HLS chip filenames and the source HF dataset
only carry a grid index (chip_ROW_COL), not per-frame dates.

Usage (on cuda2, from repo root):
    python3 export_chips_kml.py              # full dataset
    python3 export_chips_kml.py --study-only # 500-chip study set
Output:
    chip_locations.kml
"""

import argparse
import csv
import glob
from pathlib import Path

import rasterio
from pyproj import Transformer

CHIPS_DIR = Path("multi_tile_generalization/training_chips")
RESULTS_CSV = Path("multi_tile_generalization/block_masking_study/outputs/results_600M.csv")
OUT_KML = Path("chip_locations.kml")


def get_study_chip_ids():
    ids = set()
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["chip"])
    return ids


def get_center_latlon(tif_path, transformer):
    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
        cx = (left + right) / 2
        cy = (bottom + top) / 2
        lon, lat = transformer.transform(cx, cy)
    return lat, lon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-only", action="store_true",
                        help="export only the chips listed in results_600M.csv")
    args = parser.parse_args()

    all_merged_files = sorted(glob.glob(str(CHIPS_DIR / "*_merged.tif")))
    if args.study_only:
        study_ids = get_study_chip_ids()
        print(f"Found {len(study_ids)} unique chips in {RESULTS_CSV}")
        merged_files = [f for f in all_merged_files if Path(f).name in study_ids]
        print(f"Matched {len(merged_files)} of those to files on disk "
              f"(out of {len(all_merged_files)} total *_merged.tif files present)")
    else:
        merged_files = all_merged_files
        print(f"Exporting all {len(merged_files)} *_merged.tif chips in {CHIPS_DIR}")

    if not merged_files:
        raise SystemExit("No matching chip files found -- check CHIPS_DIR / RESULTS_CSV")

    with rasterio.open(merged_files[0]) as src:
        src_crs = src.crs
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    placemarks = []
    for f in merged_files:
        chip_id = Path(f).stem.replace("_merged", "")
        lat, lon = get_center_latlon(f, transformer)
        placemarks.append(
            f"""    <Placemark>
      <name>{chip_id}</name>
      <Point>
        <coordinates>{lon:.6f},{lat:.6f},0</coordinates>
      </Point>
    </Placemark>"""
        )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Prithvi Chip Locations ({len(placemarks)} chips)</name>
{chr(10).join(placemarks)}
  </Document>
</kml>
"""
    OUT_KML.write_text(kml)
    print(f"Wrote {len(placemarks)} placemarks to {OUT_KML}")


if __name__ == "__main__":
    main()
