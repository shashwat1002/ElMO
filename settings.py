import torch

REMOVAL_CHARACTERS = [
    "&",
    "@",
    "#",
    "\"",
    "\n",
    "_",
    ",",
    "{",
    "}",
    "<",
    ">",
    "[",
    "]",
    "(",
    ")"
]

SPECIAL_TOKEN_DICT = {
    "<eos>": 0,
    "<sos>": 1,
    "<pad>": 2,
    "<unk>": 3
}

SPECIAL_TOKENS = [
    key for key in SPECIAL_TOKEN_DICT
]

N = 4

HIDDEN_SIZE = 50

DEVICE = torch.device("cpu")

# MIN_FREQ = 3
MIN_FREQ = 5 #do this if vocab index out of range

TRAIN_FILE_PATH = "data/anlp-assgn2-data/yelp-subset.train.csv"
TEST_FILE_PATH = "data/anlp-assgn2-data/yelp-subset.test.csv"
VALIDATION_FILE_PATH = "data/anlp-assgn2-data/yelp-subset.dev.csv"

NUM_EPOCHS = 5
BATCH_SIZE = 100