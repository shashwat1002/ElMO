import torch
from torch.nn import Module
from icecream import ic
from data import Language, DatasetYELP
from settings import *
from torch.utils.data import DataLoader
from torch import nn


TRAIN_MODES = {
    "train": 0,
    "fine-tune": 1,
    "test": 2
}

class ELMO(torch.nn.Module):

    def __init__(self, vocab, glove_embeddings_for_vocab, hidden_size):
        super().__init__()
        self.vocab = vocab
        self.glove_dim = hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = len(vocab)


        self.embedding_layer = nn.Embedding.from_pretrained(glove_embeddings_for_vocab)

        self.lstm_forward_1 = nn.LSTM(self.glove_dim, hidden_size, batch_first=True)
        self.lstm_forward_2 = nn.LSTM(hidden_size, self.hidden_size, batch_first=True)

        self.lstm_backward_1 = nn.LSTM(self.glove_dim, hidden_size, batch_first=True)
        self.lstm_backward_2 = nn.LSTM(hidden_size, self.glove_dim, batch_first=True)

        self.score_layer = nn.Linear(hidden_size, self.vocab_size)

        self.trainable_weights_for_task = nn.Parameter(torch.randn((1, 3)).to(DEVICE))

        self.mode_t = TRAIN_MODES["train"]
        # this is the default

        self.yelp_classifier = nn.Linear(hidden_size*2, 5)


    def fine_tune_mode(self):
        self.mode_t = TRAIN_MODES["fine-tune"]

    def forward_all_info(self, batch):
        # forward for the Language Modelling task specifically
        # batch is assumed to be: batch_size x sequence_length

        batch_size = batch.size()[0]

        input_embeddings = self.embedding_layer(batch)

        hidden1_for_init = torch.zeros([1, batch_size, self.hidden_size]).to(DEVICE)
        hidden1_back_init =torch.zeros([1, batch_size, self.hidden_size]).to(DEVICE)
        hidden2_for_init =torch.zeros([1, batch_size, self.hidden_size]).to(DEVICE)
        hidden2_back_init = torch.zeros([1, batch_size, self.hidden_size]).to(DEVICE)

        c_0_for1 = torch.zeros([1, batch_size, self.hidden_size]).to(DEVICE)
        c_0_for2 = torch.zeros([1, batch_size, self.hidden_size]).to(DEVICE)
        c_0_back1 = torch.zeros([1, batch_size, self.hidden_size]).to(DEVICE)
        c_0_back2 = torch.zeros([1, batch_size, self.hidden_size]).to(DEVICE)



        hidden1_for, (_, _) = self.lstm_forward_1(input_embeddings, (hidden1_for_init, c_0_for1))
        hidden_final_for, (_, _) = self.lstm_forward_2(input_embeddings, (hidden2_for_init, c_0_for2))

        input_rev_embeddings = torch.flip(input_embeddings, [1])

        hidden1_back, (_, _) = self.lstm_backward_1(input_rev_embeddings, (hidden1_back_init, c_0_back1))
        hidden_final_back, (_, _) = self.lstm_backward_2(input_rev_embeddings, (hidden2_back_init, c_0_back2))
        # hidden_final_back: batch_size x sequence_length x hidden_size

        hidden1_back_rev = torch.flip(hidden1_back, [1])
        hidden_final_back_rev = torch.flip(hidden_final_back, [1])


        return (hidden_final_for, hidden1_for, hidden_final_back_rev, hidden1_back_rev)

    def forward_pretrain(self, batch):
        hidden2_for, hidden1_for, hidden2_back, hidden1_back = self.forward_all_info(batch)
        score_for, score_back = self.score_layer(hidden2_for), self.score_layer(hidden2_back)
        return (score_for, score_back)

    def forward_finetune_task(self, batch):
        hidden2_for, hidden1_for, hidden2_back, hidden1_back = self.forward_all_info(batch)
        # dimensions of all: batch_size x sequence_length x hidden_size

        ic(batch.size())
        input_embeddings = self.embedding_layer(batch)
        batch_size, sequence_length, hidden_size = input_embeddings.size()
        ic(input_embeddings.size())

        hidden2 = torch.cat((hidden2_for, hidden2_back), 2)
        hidden1 = torch.cat((hidden1_for, hidden1_back), 2)
        ic(hidden1.size())

        input_f = input_embeddings.repeat((1, 1, 2))
        ic(input_f.size())

        stacked = torch.stack((hidden1, hidden2, input_f),  2)
        # batch_size x sequence_length x 3 x (2*hidden_size)
        ic(stacked.size())

        embeddings_f = torch.matmul(self.trainable_weights_for_task, stacked)
        embeddings_f_squeezed = torch.squeeze(embeddings_f, dim=len(embeddings_f.size())-2)
        # batch_size x sequence_len x (2*hidden_size)
        ic(embeddings_f.size())
        ic(embeddings_f_squeezed.size())

        return embeddings_f

    def forward(self, batch):
        if self.mode_t == TRAIN_MODES["train"] or self.mode_t == TRAIN_MODES["test"]:
            return self.forward_pretrain(batch)
        else:
            return self.forward_finetune_task(batch)






# language = Language("data/anlp-assgn2-data/yelp-subset.train.csv",
#                     MIN_FREQ, glove_dim=HIDDEN_SIZE, glove_name="twitter.27B")

# test_dataset = DatasetYELP(
#     language, "data/anlp-assgn2-data/yelp-subset.train.csv", pad=True)
# test_dataloader = DataLoader(test_dataset, batch_size=3)

# elmodel = ELMO(language.vocab, language.get_embedding_list_intersect_vocab(), HIDDEN_SIZE).to(DEVICE)

# elmodel.mode_t = 0
# ic(elmodel((next(iter(test_dataloader))[0])))


# elmodel.mode_t = 1
# ic(elmodel((next(iter(test_dataloader))[0])))
