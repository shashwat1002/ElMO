
from torchtext import vocab
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torch.utils.data import Dataset, DataLoader
from icecream import ic
from settings import *
import re
import random
import csv


TOKENIZER = get_tokenizer('basic_english')

MIN_FREQ = 1


class Language:
    # class to keep information like vocabulary

    def __init__(self, file_path, min_freq, glove_dim=50, glove_name="twitter.27B"):

        self.glove_dim = glove_dim
        self.glove_name = glove_name
        sentence_list = []

        with open(file_path, "r") as data_file:
            dict_reader_obj = csv.DictReader(data_file, ["label", "text"])
            for i, entry in enumerate(dict_reader_obj):

                if i == 0:
                    # don't read first line
                    continue
                sentence = entry["text"]

                for character in REMOVAL_CHARACTERS:
                    sentence = sentence.replace(character, " ")
                sentence_list.append(sentence)

        sentence_list_tokenized = [
            TOKENIZER(sentence) for sentence in sentence_list]
        self.vocab = vocab.build_vocab_from_iterator(sentence_list_tokenized,
                                                     min_freq=min_freq,
                                                     specials=SPECIAL_TOKENS,
                                                     special_first=True)

        # use this index for OOV
        # THIS LINE ONLY WORKS IF special_first=True
        self.vocab.set_default_index(SPECIAL_TOKEN_DICT["<unk>"])

        self.sentence_list = sentence_list

        self.glove_layer = None

        self.initialize_GLoVE()

        self._intersect_vocab(self.glove_layer.stoi)

    def initialize_GLoVE(self):
        ic(self.glove_dim)
        self.glove_layer = GloVe(name=self.glove_name, dim=self.glove_dim)

    def _intersect_vocab(self, vocab_in):
        # find intersection of data vocab
        # and pretrained vocab
        vocab_len = self.vocab.__len__()

        intersection_token_list = []
        # we definitely want the special tokens in:

        num_specials = len(SPECIAL_TOKENS)

        for i in range(vocab_len):
            if i < num_specials:
                # assuming specials are at the start
                intersection_token_list.append(self.vocab.lookup_token(i))
            elif vocab_in.__contains__(self.vocab.lookup_token(i)):
                # intersection
                intersection_token_list.append(self.vocab.lookup_token(i))

        # we have the intersection list
        # we generate the vocabulary from this

        self.vocab = vocab.build_vocab_from_iterator([[token] for token in intersection_token_list],
                                                     specials=SPECIAL_TOKENS,
                                                     special_first=True)
        self.vocab.set_default_index(SPECIAL_TOKEN_DICT["<unk>"])

    def get_embedding_list_intersect_vocab(self):
        return self.glove_layer.get_vecs_by_tokens(self.vocab.get_itos()).to(DEVICE)


class DatasetYELP(Dataset):

    def __init__(self, language_obj, file_path, pad=False):

        sentence_set = []
        label_set = []
        self.language = language_obj

        with open(file_path, "r") as data_file:
            dict_reader_obj = csv.DictReader(data_file, ["label", "text"])
            for i, entry in enumerate(dict_reader_obj):

                if i == 0:
                    # don't read first line: it has fieldnames
                    continue

                sentence = entry["text"]

                for character in REMOVAL_CHARACTERS:
                    sentence = sentence.replace(character, " ")
                sentence_set.append(sentence)
                label_set.append(int(entry["label"]))

        self.language_obj = language_obj
        self.tokenized_sentences = []

        length_longest = 0
        self.pad = pad

        for sentence in sentence_set:
            sentence = "<sos> " + sentence
            tokenized_sentence = TOKENIZER(sentence)
            if len(tokenized_sentence) > length_longest:
                length_longest = len(tokenized_sentence)

            self.tokenized_sentences.append(tokenized_sentence)

        self.label_set = label_set
        # ic(length_longest)

        for i, sentence_tokenized in enumerate(self.tokenized_sentences):
            # for padding
            if pad:
                self.tokenized_sentences[i] = sentence_tokenized + [
                    "<eos>" for i in range(length_longest-len(sentence_tokenized))]
                # ic(length_longest-len(sentence_tokenized), len(["<eos>" * (length_longest-len(sentence_tokenized))]))
            else:
                sentence_tokenized.append("<eos>")

        self.words_as_indices_sentences = []

        for sentence_tokenized in self.tokenized_sentences:
            sentence_as_index_list = self.language_obj.vocab.lookup_indices(
                sentence_tokenized)
            self.words_as_indices_sentences.append(sentence_as_index_list)

    def __len__(self):
        return len(self.words_as_indices_sentences)

    def __getitem__(self, index):

        sentence_index_list = self.words_as_indices_sentences[index]
        # if len(sentence_index_list) > 200 and self.pad:
        #     sentence_index_list = sentence_index_list[:200]
        return torch.tensor(sentence_index_list, device=DEVICE), torch.tensor(self.label_set[index])


# language = Language("data/anlp-assgn2-data/yelp-subset.train.csv",
#                     MIN_FREQ, glove_dim=HIDDEN_SIZE, glove_name="twitter.27B")

# test_dataset = DatasetYELP(
#     language, "data/anlp-assgn2-data/yelp-subset.train.csv", pad=True)
# test_dataloader = DataLoader(test_dataset, batch_size=3)


# ic(next(iter(test_dataloader)))
# ic(test_dataset.language.get_embedding_list_intersect_vocab().size())
# ic(test_dataset.language.get_embedding_list_intersect_vocab()[0:5, :])

# ic(next(iter(test_dataloader)))
