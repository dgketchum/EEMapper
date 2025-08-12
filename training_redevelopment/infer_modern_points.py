import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from combined_classifier import CombinedClassifier
from combined_dataset import PreloadedDataset, load_and_preprocess_data
from training_redevelopment import NUMBER_CLASSES, NUMBER_WORKERS
from training_redevelopment.get_files import get_files


def load_model_and_metadata(checkpoint_dir, checkpoint_filename=None):
    if not checkpoint_filename:
        ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        latest_ckpt = max([os.path.join(checkpoint_dir, f) for f in ckpts], key=os.path.getmtime)
        checkpoint_filename = latest_ckpt
    else:
        checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_filename)

    datestamp = os.path.basename(checkpoint_filename).replace('model_', '').replace('.ckpt', '')
    checkpoint_metadata = os.path.join(checkpoint_dir, f'metadata_{datestamp}.json')

    with open(checkpoint_metadata, 'r') as f:
        metadata = json.load(f)

    for cat, map_dict in metadata['categorical_mappings'].items():
        metadata['categorical_mappings'][cat] = {int(k): v for k, v in map_dict.items()}

    model = CombinedClassifier.load_from_checkpoint(checkpoint_filename)
    model.eval()
    return model, metadata


def infer_data(model, metadata, bands_dir, ndvi_dir, all_points_gdf, out_dir, sample=None, debug=False,
               batch_size=64, data_type='past', states=None, filter_points=False):
    files = get_files(bands_dir, ndvi_dir, sample)
    files.sort()

    fids = ['_'.join(os.path.basename(f[0]).split('_')[:2]) for f in files]

    if states:
        fids = [fid for fid in fids if fid[:2] in states]
        files = [f for f in files if f[0][:2] in states]

    data = load_and_preprocess_data(files, bands_dir, ndvi_dir,
                                    metadata['all_features'], metadata['categorical_mappings'],
                                    num_workers=NUMBER_WORKERS, debug=debug)

    dataset = PreloadedDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=NUMBER_WORKERS, pin_memory=True)

    print(f'{len(dataset)} inference points')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inferring {data_type} data"):
            x, y = batch
            x = [t.to(device) for t in x]
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    results_df = pd.DataFrame({'FID': fids, 'NEW_CLASS': predictions})

    points_gdf = all_points_gdf[all_points_gdf['FID'].isin(fids)]
    merged_gdf = points_gdf.merge(results_df, on='FID')

    if NUMBER_CLASSES == 2:
        merged_gdf.loc[merged_gdf['POINT_TYPE'] > 0, 'POINT_TYPE'] = 1

    merged_gdf['static'] = np.where(((merged_gdf['POINT_TYPE'] == 0) & (merged_gdf['NEW_CLASS'] == 0)) |
                                    ((merged_gdf['POINT_TYPE'] > 0) & (merged_gdf['NEW_CLASS'] == 1)), 1, 0)

    if data_type == 'modern':
        uncertain_points = merged_gdf.loc[merged_gdf['static'] == 0].copy()
        uncertain_points.to_file(os.path.join(out_dir, f'uncertain_points.shp'))

    if filter_points:
        merged_gdf = merged_gdf.loc[merged_gdf['static'] == 1]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for state in merged_gdf['STUSPS'].unique():

        if states is not None and state not in states:
            continue

        state_gdf = merged_gdf[merged_gdf['STUSPS'] == state]
        state_gdf = state_gdf[['FID', 'POINT_TYPE', 'NEW_CLASS', 'YEAR', 'NEW_YEAR', 'static', 'geometry']]
        out_shp = os.path.join(out_dir, f'{state}_inferred_{data_type}_points.shp')
        state_gdf.to_file(out_shp)
        print(out_shp)


def infer_past_data(checkpoint_dir, past_bands_dir, past_ndvi_dir, all_points_gdf, out_dir, **kwargs):
    model, metadata = load_model_and_metadata(checkpoint_dir, kwargs.pop('checkpoint_filename'))
    infer_data(model, metadata, past_bands_dir, past_ndvi_dir, all_points_gdf, out_dir, data_type='past', **kwargs)


def infer_future_data(checkpoint_dir, modern_bands_dir, modern_ndvi_dir, all_points_gdf, out_dir, **kwargs):
    model, metadata = load_model_and_metadata(checkpoint_dir, kwargs.pop('checkpoint_filename'))
    infer_data(model, metadata, modern_bands_dir, modern_ndvi_dir, all_points_gdf, out_dir, data_type='future',
               **kwargs)


if __name__ == '__main__':

    root_ = '/media/research/IrrigationGIS'
    if not os.path.exists(root_):
        root_ = '/home/dgketchum/data/IrrigationGIS'

    data_ = '/data/ssd2/irrmapper/states'
    past_bands_dir_ = os.path.join(data_, 'bands', 'past_processed')
    past_ndvi_dir_ = os.path.join(data_, 'timeseries', 'past_processed')
    modern_bands_dir_ = os.path.join(data_, 'bands', 'modern_processed')
    modern_ndvi_dir_ = os.path.join(data_, 'timeseries', 'modern_processed')
    checkpoint_dir_ = os.path.join(data_, 'combined_classifier')

    extracts_ = os.path.join(root_, 'irrmapper', 'EE_extracts', 'point_shp')
    points_shp_dir_ = os.path.join(extracts_, 'state_wgs_mgrs')
    out_dir_ = os.path.join(extracts_, 'state_wgs_inferred')

    shp_files_ = [os.path.join(points_shp_dir_, f) for f in os.listdir(points_shp_dir_) if f.endswith('.shp')]
    all_points_gdf_ = pd.concat([gpd.read_file(shp) for shp in shp_files_], ignore_index=True)

    # states_ = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
    states_ = ['KS', 'ND', 'NE', 'OK', 'SD', 'TX']

    kwargs_ = {

        # western 17 states
        'checkpoint_filename': 'model_20250811_181538.ckpt',

        # western 11 states
        # 'checkpoint_filename': 'model_20250811_124746.ckpt',

        'states': states_,
        'sample': None,
        'debug': False,
        'batch_size': 128,
    }

    # infer_past_data(checkpoint_dir_, past_bands_dir_, past_ndvi_dir_,
    #                 all_points_gdf_, out_dir_, **kwargs_)

    infer_future_data(checkpoint_dir_, modern_bands_dir_, modern_ndvi_dir_,
                      all_points_gdf_, out_dir_, **kwargs_)

# ========================= EOF ====================================================================
