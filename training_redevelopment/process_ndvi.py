import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm


def process_file(file_path, out_dir, new_year=False):
    df = pd.read_csv(file_path)
    df.set_index('FID', inplace=True)

    for fid, row in df.iterrows():

        if new_year:
            year = row['NEW_YEAR']
        else:
            year = row['YEAR']

        idx = pd.DatetimeIndex(pd.date_range(f'{year}-01-01', f'{year}-12-31'), freq='D')
        data = {'ndvi': row.drop(['YEAR', 'NEW_YEAR', 'POINT_TYPE', 'MGRS_TILE', 'STUSPS']).values}

        pt_type = row['POINT_TYPE']
        mgrs = row['MGRS_TILE']
        state = row['STUSPS']

        out_file = os.path.join(out_dir, f'{fid}_{year}_{state}_{mgrs}_{pt_type}.parquet')
        if os.path.exists(out_file):
            continue

        dts = [pd.to_datetime(x.split('_')[-1]) for x in row.drop(['YEAR', 'NEW_YEAR', 'POINT_TYPE',
                                                                   'MGRS_TILE', 'STUSPS']).index]
        s = pd.Series(data['ndvi'], index=dts)

        if s.isnull().all():
            continue

        s[s < 0.05] = np.nan
        s = s.sort_index()
        s = s.resample('D').max()
        s = s.reindex(idx)

        if len(s.index) == 366:
            s = s[~((s.index.month == 2) & (s.index.day == 29))]

        s = s.astype(float).ffill().bfill()
        s.fillna(0, inplace=True)

        s = pd.DataFrame(savgol_filter(s, 21, 3), index=s.index, columns=[fid])
        s.to_parquet(out_file)


def process_ndvi_time_series(in_dir, out_dir, modern_update=False, num_workers=None):
    file_list = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.csv')]

    if num_workers == 1:
        for f in tqdm(file_list):
            process_file(f, out_dir)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_file, f, out_dir, modern_update) for f in file_list]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()


if __name__ == '__main__':

    in_dir_past = '/data/ssd2/irrmapper/states/timeseries/ndvi_past/'
    out_dir_past = '/data/ssd2/irrmapper/states/timeseries/past_processed/'
    process_ndvi_time_series(in_dir_past, out_dir_past, num_workers=24)

    in_dir_modern = '/data/ssd2/irrmapper/states/timeseries/ndvi_modern/'
    out_dir_modern = '/data/ssd2/irrmapper/states/timeseries/modern_processed/'
    process_ndvi_time_series(in_dir_modern, out_dir_modern, modern_update=True, num_workers=24)

# ========================= EOF ====================================================================
