import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from training_redevelopment import NUMBER_CLASSES


class PreloadedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _load_and_preprocess_worker(args):
    band_fname, ndvi_fname, bands_dir, ndvi_dir, all_features, categorical_mappings = args
    categorical_features = sorted(list(categorical_mappings.keys()))
    continuous_features = [f for f in all_features if f not in categorical_mappings]

    bands_path = os.path.join(bands_dir, band_fname)
    bands_df = pd.read_parquet(bands_path)
    bands_df.fillna(0, inplace=True)

    for col in all_features:
        if col not in bands_df.columns:
            bands_df[col] = 0

    for cat, mapping in categorical_mappings.items():
        bands_df[cat] = bands_df[cat].map(mapping).fillna(0).astype(int)

    continuous_data = bands_df[continuous_features].values.astype(np.float32).squeeze()
    categorical_data = bands_df[categorical_features].values.astype(np.int64).squeeze()

    ndvi_path = os.path.join(ndvi_dir, ndvi_fname)
    ndvi_df = pd.read_parquet(ndvi_path)
    ndvi_series = ndvi_df.values.astype(np.float32).T

    label = int(band_fname.split('_')[-1].split('.')[0])
    if NUMBER_CLASSES == 2 and label > 0:
        label = 1
    elif NUMBER_CLASSES == 4 and label > 3:
        label = 1

    return (continuous_data, categorical_data, ndvi_series), label


def load_and_preprocess_data(file_list, bands_dir, ndvi_dir, all_features, categorical_mappings, num_workers=8,
                             debug=False):
    args_list = [(band_fname, ndvi_fname, bands_dir, ndvi_dir, all_features, categorical_mappings)
                 for band_fname, ndvi_fname in file_list]

    args_list.sort()

    if debug:
        print("Running pre-loading in debug (sequential) mode.")
        results = []
        for args in tqdm(args_list, desc="Loading and preprocessing data (debug)"):
            results.append(_load_and_preprocess_worker(args))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(_load_and_preprocess_worker, args_list), total=len(args_list),
                                desc="Loading and preprocessing data"))

    processed_data = []
    for (continuous, categorical, ndvi), label in results:
        if continuous is not None:
            continuous_t = torch.from_numpy(continuous)
            categorical_t = torch.from_numpy(categorical)
            ndvi_t = torch.from_numpy(ndvi)
            label_t = torch.tensor(label, dtype=torch.long)
            processed_data.append(((continuous_t, categorical_t, ndvi_t), label_t))

    return processed_data


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
