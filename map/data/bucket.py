import os

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/dgketchum/ssebop-montana-78fae6d904e1.json'
client = storage.Client()


def get_bucket_contents(bucket_name, glob=None):
    dct = {}
    empty = []
    for blob in client.list_blobs(bucket_name, prefix=glob):
        dirname = os.path.dirname(blob.name)
        b_name = os.path.basename(blob.name)
        if blob.size == 20:
            empty.append(blob.name)
            continue
        size = blob.size / 1e6
        if dirname not in dct.keys():
            dct[dirname] = [(b_name, size)]
        else:
            dct[dirname].append((b_name, size))
    for k, v in dct.items():
        print(k, sum([x[1] for x in v]))
    return dct


if __name__ == '__main__':
    bucket = 'ts_data'
    get_bucket_contents(bucket)
# ========================= EOF ====================================================================
