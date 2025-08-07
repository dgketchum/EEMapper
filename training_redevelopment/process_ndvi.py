import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm


def process_ndvi_time_series(in_dir, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for f in tqdm(os.listdir(in_dir)):
        if not f.endswith('.csv'):
            continue

        df = pd.read_csv(os.path.join(in_dir, f))
        df.set_index('FID', inplace=True)

        for fid, row in df.iterrows():
            year = row['YEAR']

            idx = pd.DatetimeIndex(pd.date_range(f'{year}-01-01', f'{year}-12-31'), freq='D')
            data = {'ndvi': row.drop(['YEAR', 'POINT_TYPE', 'MGRS_TILE', 'STUSPS']).values}

            pt_type = row['POINT_TYPE']
            mgrs = row['MGRS_TILE']
            state = row['STUSPS']

            out_file = os.path.join(out_dir, f'{fid}_{year}_{state}_{mgrs}_{pt_type}.parquet')
            if os.path.exists(out_file):
                continue

            dts = [pd.to_datetime(x.split('_')[-1]) for x in row.drop(['YEAR', 'POINT_TYPE', 'MGRS_TILE', 'STUSPS']).index]
            s = pd.Series(data['ndvi'], index=dts)
            if s.isnull().all():
                continue
            s[s < 0.05] = np.nan
            s = s.resample('D').max()
            s = s.reindex(idx)
            if len(s.index) == 366:
                s = s[~((s.index.month == 2) & (s.index.day == 29))]
            s = s.astype(float).ffill().bfill()
            s.fillna(0, inplace=True)
            s = pd.DataFrame(savgol_filter(s, 7, 3), index=s.index, columns=[fid])
            s.to_parquet(out_file)

if __name__ == '__main__':
    in_dir_past = '/data/ssd2/irrmapper/ndvi/past_training/'
    out_dir_past = '/data/ssd2/irrmapper/ndvi/past_processed/'
    process_ndvi_time_series(in_dir_past, out_dir_past)

    in_dir_modern = '/data/ssd2/irrmapper/ndvi/modern_update/'
    out_dir_modern = '/data/ssd2/irrmapper/ndvi/modern_processed/'
    process_ndvi_time_series(in_dir_modern, out_dir_modern)

# ========================= EOF ====================================================================
