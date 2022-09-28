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

DEVICE = torch.device("cuda")

MIN_FREQ = 3