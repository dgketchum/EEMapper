import os
import random
from training_redevelopment import RANDOM_SEED

def get_files(bands, ndvi, sample_n=None):
    ndvi_files = {f for f in os.listdir(ndvi) if f.endswith('.parquet')}
    all_files = []
    for f in os.listdir(bands):
        if not f.endswith('.parquet'):
            continue
        parts = f.split('_')
        ndvi_f = '_'.join(parts[:3] + parts[4:])
        if ndvi_f not in ndvi_files:
            ndvi_f = '_'.join(parts[:2] + parts[3:])
        if ndvi_f in ndvi_files:
            all_files.append((f, ndvi_f))

    all_files.sort()

    if sample_n:
        random.seed(RANDOM_SEED)
        sample = random.sample(all_files, sample_n)
        return sample

    return all_files


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
