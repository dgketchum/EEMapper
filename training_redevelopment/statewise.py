import os
from subprocess import check_call
from datetime import datetime

import fiona
import geopandas as gpd
import pandas as pd
from map.call_ee import is_authorized, request_band_extract

#
# ALL_STATES = TARGET_STATES + E_STATES

home = os.path.expanduser("~")
conda = os.path.join(home, "miniconda3", "envs")
if not os.path.exists(conda):
    conda = conda.replace("miniconda3", "miniconda")
EE = "/home/dgketchum/miniconda3/envs/met/bin/earthengine"
GS = "/home/dgketchum/google-cloud-sdk/bin/gsutil"

OGR = "/usr/bin/ogr2ogr"

AEA = (
    "+proj=aea +lat_0=40 +lon_0=-96 +lat_1=20 +lat_2=60 +x_0=0 +y_0=0 +ellps=GRS80 "
    "+towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
)
WGS = "+proj=longlat +datum=WGS84 +no_defs"

os.environ["GDAL_DATA"] = "miniconda3/envs/gcs/share/gdal/"

DRYLAND_STATES = ["CO", "ID", "MT", "OR", "WA"]

TRAINING_DATA = {
    "AZ": "users/dgketchum/bands/state/AZ_24NOV2021",
    "CA": "users/dgketchum/bands/state/CA_14NOV2021",
    "CO": "users/dgketchum/bands/state/CO_10NOV2021",
    "ID": "users/dgketchum/bands/state/ID_10NOV2021",
    "KS": "users/dgketchum/bands/state/KS_7NOV2021",
    "MT": "users/dgketchum/bands/state/MT_15NOV2021",
    "ND": "users/dgketchum/bands/state/ND_7NOV2021",
    "NE": "users/dgketchum/bands/state/NE_7NOV2021",
    "NM": "users/dgketchum/bands/state/NM_7NOV2021",
    "NV": "users/dgketchum/bands/state/NV_7NOV2021",
    "OK": "users/dgketchum/bands/state/OK_7NOV2021",
    "OR": "users/dgketchum/bands/state/OR_22NOV2021",
    "SD": "users/dgketchum/bands/state/SD_7NOV2021",
    "TX": "users/dgketchum/bands/state/TX_7NOV2021",
    "UT": "users/dgketchum/bands/state/UT_10NOV2021",
    "WA": "users/dgketchum/bands/state/WA_10NOV2021",
    "WY": "users/dgketchum/bands/state/WY_7NOV2021",
}


def to_geographic(in_dir, out_dir, states, mgrs_path, n_samples=None):
    df_list = []
    for state in states:
        in_shp = [
            os.path.join(in_dir, x)
            for x in os.listdir(in_dir)
            if x.endswith(".shp") and state in x
        ]
        dates = []
        for s in in_shp:
            try:
                d = datetime.strptime(s.split("_")[-1].split(".")[0], "%d%b%Y")
                dates.append(d)
            except ValueError:
                continue
        latest = in_shp[dates.index(max(dates))]
        mgrs = gpd.read_file(mgrs_path)

        points = gpd.read_file(latest)
        points = points.sjoin(mgrs, how="inner")
        points = points.to_crs(epsg=4326)
        points.drop(columns=["index_right"], inplace=True)
        points["STUSPS"] = state

        if n_samples:
            points = points.groupby("POINT_TYPE", group_keys=False).apply(
                lambda x: x.sample(n=n_samples)
            )
            points.index = list(range(points.shape[0]))

        df_list.append(points)

    df = pd.concat(df_list, ignore_index=True)
    df["FID"] = ["%s_%s" % (row["STUSPS"], str(i).zfill(6)) for i, row in df.iterrows()]

    out_shp = os.path.join(out_dir, "master_training_points.shp")
    df.to_file(out_shp)
    print(out_shp)
    return out_shp


def push_points_to_asset(_dir, shapefile, bucket):
    shp_name = os.path.basename(shapefile).replace(".shp", "")
    local_files = [
        os.path.join(_dir, "{}.{}".format(shp_name, ext))
        for ext in ["shp", "prj", "shx", "dbf"]
    ]
    bucket = os.path.join(bucket, "state_points")
    bucket_files = [
        os.path.join(bucket, "{}.{}".format(shp_name, ext))
        for ext in ["shp", "prj", "shx", "dbf"]
    ]
    for lf, bf in zip(local_files, bucket_files):
        cmd = [GS, "cp", lf, bf]
        check_call(cmd)

    asset_id = os.path.basename(bucket_files[0]).split(".")[0]
    ee_dst = "users/dgketchum/points/state_redev/{}".format(asset_id)
    cmd = [EE, "upload", "table", "-f", "--asset_id={}".format(ee_dst), bucket_files[0]]
    check_call(cmd)
    print(asset_id, bucket_files[0])


def get_bands(pts_dir, glob, out_glob, state, southern=False):
    pts = os.path.join(pts_dir, "points_{}_{}.shp".format(state, glob))
    with fiona.open(pts, "r") as src:
        years = list(set([x["properties"]["YEAR"] for x in src]))
    print("get bands", state)
    pts = "users/dgketchum/points/state/points_{}_{}".format(state, glob)
    geo = "users/dgketchum/boundaries/{}".format(state)
    file_ = "bands_{}_{}".format(state, out_glob)
    request_band_extract(
        file_,
        pts,
        region=geo,
        years=years,
        filter_bounds=True,
        buffer=1e5,
        southern=southern,
        diagnose=False,
    )


if __name__ == "__main__":
    is_authorized()
    _bucket = "gs://wudr"
    root = "/media/research/IrrigationGIS/irrmapper"
    if not os.path.exists(root):
        root = "/home/dgketchum/data/IrrigationGIS/irrmapper"

    pt = os.path.join(root, "EE_extracts/point_shp")
    pt_wgs = os.path.join(pt, "state_wgs_mgrs")
    pt_aea = os.path.join(pt, "state_aea")

    extracts = os.path.join(root, "EE_extracts")
    to_concat = os.path.join(extracts, "to_concatenate/state")
    conctenated = os.path.join(extracts, "concatenated/state")
    imp_json = os.path.join(extracts, "variable_importance", "statewise")

    coll = "users/dgketchum/IrrMapper/IrrMapper_AE"
    # coll = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp_'
    tables = "users/dgketchum/bands/state"
    mgrs = "/media/research/IrrigationGIS/boundaries/mgrs/mgrs_aea.shp"

    states = ["AZ", "CA", "CO", "ID", "MT", "NM", "NV", "OR", "UT", "WA", "WY"]
    shp = to_geographic(pt_aea, pt_wgs, states=states, mgrs_path=mgrs, n_samples=500)
    push_points_to_asset(pt_wgs, shp, bucket=_bucket)
