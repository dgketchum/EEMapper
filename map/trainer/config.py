PROJECT = 'GEE-IrrMapper'
BUCKET = 'ee-irrigation-mapping'
REMOTE_OR_LOCAL = 'remote'
FOLDER = 'models/{}-0.01wd-ndvi-2mparams-final-split'.format(REMOTE_OR_LOCAL)
JOB_DIR = 'gs://' + BUCKET + '/' + FOLDER + '/trainer'
MODEL_DIR = JOB_DIR + '/model'
LOGS_DIR = JOB_DIR + '/logs'

DATA_BUCKET = ''
TRAIN_BASE = 'train-data-july23/'
TEST_BASE = 'test-data-july23/'
TEST_SIZE = 8673  # 19446

if REMOTE_OR_LOCAL == 'remote':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 32

EPOCHS = 300
STEPS_PER_EPOCH = 400
BUFFER_SIZE = 10
N_CLASSES = 5
