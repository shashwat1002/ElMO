from model import ELMO
from torch import nn
from data import Language, DatasetYELP, DataLoader
from settings import *
from icecream import ic

def train_epoch_LM(model, loss_fun, dataloader, optimizer):

    model.train()
    model.mode_t = 0
    for batch in dataloader:

        batch = batch[0]
        (forward_scores, backward_scores) = model(batch)
        batch_size, sequence_length, vocab_size = forward_scores.size()
        forward_scores_slice = forward_scores[:, :sequence_length-1, :]
        backward_scores_slice = backward_scores[:, :sequence_length-1, :]
        # ic(scores.shape, batch.shape)
        # ic(scores.view(scores.shape[0]*scores.shape[1], -1).shape, batch.view(scores.shape[0]*scores.shape[1]).shape)
        # loss = loss_fun(scores.view(scores.shape[0]*scores.shape[1], -1), batch[:, 1:].view(scores.shape[0]*scores.shape[1]))
        # ic(scores.shape, batch[:, 1:].shape)
        # ic(scores.view(-1, scores.shape[2]))
        # ic(batch[:, 1:].contiguous().view(-1))
        batch_for_ground_truth = batch[:, 1:]
        batch_back_ground_truth = batch[:, :-1]

        loss_for = loss_fun(forward_scores_slice.contiguous().view(-1, forward_scores_slice.shape[2]), batch_for_ground_truth.contiguous().view(-1))
        # ic(loss_for)
        loss_back = loss_fun(backward_scores_slice.contiguous().view(-1, backward_scores_slice.shape[2]), batch_back_ground_truth.contiguous().view(-1))

        loss = loss_for + loss_back
        ic(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def train(model, optimizer, loss_fun, dataset, num_epochs, model_path, validation_dataset):

    min_loss = 9999999999
    for epoch in range(num_epochs):
        print(f"{epoch+1}")
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
        train_epoch_LM(model, loss_fun, dataloader, optimizer)
        avg_loss = validation(validation_dataset, model, dataset.language_obj)
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model, model_path) # save only when loss becomes lower
        print("-------")
        # torch.save(model, model_path)

def train_lstm_lm(file_path_train, language_obj, model_path, num_epochs, file_path_validation):


    train_dataset = DatasetYELP(
        language, file_path_train, pad=True)
    validation_dataset = DatasetYELP(language_obj, file_path_validation, pad=False)
    train_dataloader = DataLoader(test_dataset, batch_size=3)

    elmodel = ELMO(language.vocab, language.get_embedding_list_intersect_vocab(), HIDDEN_SIZE).to(DEVICE)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(elmodel.parameters(), lr=1e-3)

    train(elmodel, optimizer, loss_fun, train_dataset, num_epochs, model_path, validation_dataset)


def validation(validation_dataset, model, language_obj):


    print("------Validation------")
    model.eval()


    dataloader = DataLoader(validation_dataset, batch_size=1)

    dataset_len = len(dataset)
    total_loss = 0.0

    loss_fun = nn.CrossEntropyLoss()

    for batch in dataloader:
        batch = batch[0]
        (forward_scores, backward_scores) = model(batch)
        batch_size, sequence_length, vocab_size = forward_scores.size()
        forward_scores_slice = forward_scores[:, :sequence_length-1, :]
        backward_scores_slice = backward_scores[:, :sequence_length-1, :]

        batch_for_ground_truth = batch[:, 1:]
        batch_back_ground_truth = batch[:, :-1]

        loss_for = loss_fun(forward_scores_slice.contiguous().view(-1, forward_scores_slice.shape[2]), batch_for_ground_truth.contiguous().view(-1))
        # ic(loss_for)
        loss_back = loss_fun(backward_scores_slice.contiguous().view(-1, backward_scores_slice.shape[2]), batch_back_ground_truth.contiguous().view(-1))

        loss = loss_for + loss_back
        total_loss += loss.item()
        ic(loss)
    return total_loss / dataset_len



language = Language("data/anlp-assgn2-data/yelp-subset.train.csv",
                    MIN_FREQ, glove_dim=HIDDEN_SIZE, glove_name="twitter.27B")

test_dataset = DatasetYELP(
    language, "data/anlp-assgn2-data/yelp-subset.train.csv", pad=True)
test_dataloader = DataLoader(test_dataset, batch_size=3)

elmodel = ELMO(language.vocab, language.get_embedding_list_intersect_vocab(), HIDDEN_SIZE).to(DEVICE)

elmodel.mode_t = 0
# ic(elmodel((next(iter(test_dataloader))[0])))


# elmodel.mode_t = 1
# ic(elmodel((next(iter(test_dataloader))[0])))

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(elmodel.parameters(), lr=1e-3)

# train_epoch_LM(elmodel, loss_fun, test_dataloader, optimizer)

train_lstm_lm(TRAIN_FILE_PATH, language, "elmo.pth", NUM_EPOCHS, VALIDATION_FILE_PATH)

