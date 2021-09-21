import os

BASE_PATH = "/workspace/datasets/imagenet/ILSVRC2015"

IMAGES_PATH = os.path.sep.join([BASE_PATH, "Data/CLS-LOC"])
IMAGESETS_PATH = os.path.sep.join([BASE_PATH, "ImageSets/CLS-LOC"])
DEVKIT_PATH = os.path.sep.join([BASE_PATH, "devkit/data"])

# Path to map_clsloc.txt that maps class labels to integers
WORD_IDS = os.path.sep.join([DEVKIT_PATH, "map_clsloc.txt"])

# Lists all basename of training images
TRAIN_LIST = os.path.sep.join([IMAGESETS_PATH, "train_cls.txt"])

# Lists all basename of validation images
VALID_LIST = os.path.sep.join([IMAGESETS_PATH, "val.txt"])
VALID_LABELS = os.path.sep.join([DEVKIT_PATH, "ILSVRC2015_clsloc_validation_ground_truth.txt"])
VALID_BLACKLIST = os.path.sep.join([DEVKIT_PATH, "ILSVRC2015_clsloc_validation_blacklist.txt"])

NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# MxNet outputs
MX_OUTPUT = "/workspace/datasets/imagenet"

TRAIN_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/train.lst"])
TRAIN_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/train.rec"])

VALID_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/val.lst"])
VALID_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/val.rec"])

TEST_MX_LIST = os.path.sep.join([MX_OUTPUT, "lists/test.lst"])
TEST_MX_REC = os.path.sep.join([MX_OUTPUT, "rec/test.rec"])

# RGB Mean
MEAN_PATH = "output/mean.json"

BATCH_SIZE = 32
NUM_DEVICES = 5
