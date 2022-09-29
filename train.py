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

    dataset_len = len(validation_dataset)
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



def train_epoch_classification(model, loss_fun, dataloader, optimizer):
    model.train()
    model.mode_t = 1

    for batch in dataloader:
        inp_batch = batch[0]
        labels = batch[1]

        final_hiddens = model(inp_batch)
        # batch_size x sequence_length x hidden_size

        sentence_representation = torch.sum(final_hiddens, dim=1)

        scores = model.yelp_classifier(sentence_representation)
        scores = scores.squeeze()
        # batch_size x 5
        ic(scores.size(), labels.size())
        loss = loss_fun(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ic(loss)


def validation_classification(model, loss_fun, validation_dataset):
    dataloader = DataLoader(validation_dataset, batch_size=1)
    model.eval()
    model.mode_t = 2

    total_loss = 0
    dataset_len = len(validation_dataset)

    for batch in dataloader:
        inp_batch = batch[0]
        labels = batch[1]

        final_hiddens = model(inp_batch)

        sentence_representation = torch.sum(final_hiddens, dim=1)
        scores = model.yelp_classifier(sentence_representation)

        loss = loss_fun(scores, labels)
        total_loss += loss.item()

    return total_loss / dataset_len


def train_classification(model, optimizer, loss_fun, dataset, num_epochs, model_path, validation_dataset):

    min_loss = 9999999999
    for epoch in range(num_epochs):
        print(f"{epoch+1}")
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
        train_epoch_classification(model, loss_fun, dataloader, optimizer)
        avg_loss = validation_classification(model, loss_fun, validation_dataset)
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model, model_path) # save only when loss becomes lower
        print("-------")
        # torch.save(model, model_path)

def train_elmo_classification(model_path, train_file_path, validation_file_path, language_object, num_epochs):

    pretrained_model = torch.load(model_path).to(DEVICE)

    try:
        classifier_layer = pretrained_model.yelp_classifier
    except AttributeError:
        pretrained_model.yelp_classifier = nn.Linear(HIDDEN_SIZE*2, 5)

    train_dataset = DatasetYELP(language_object, train_file_path, pad=True)
    validation_dataset = DatasetYELP(language_object, validation_file_path)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=pretrained_model.parameters(), lr=1e-3)

    train_classification(pretrained_model, optimizer, loss_fun, train_dataset, num_epochs, model_path, validation_dataset)




language = Language("data/anlp-assgn2-data/yelp-subset.train.csv",
                    MIN_FREQ, glove_dim=HIDDEN_SIZE, glove_name="twitter.27B")

# test_dataset = DatasetYELP(
#     language, "data/anlp-assgn2-data/yelp-subset.train.csv", pad=True)
# test_dataloader = DataLoader(test_dataset, batch_size=3)

# elmodel = ELMO(language.vocab, language.get_embedding_list_intersect_vocab(), HIDDEN_SIZE).to(DEVICE)

# elmodel.mode_t = 0
# # ic(elmodel((next(iter(test_dataloader))[0])))


# # elmodel.mode_t = 1
# # ic(elmodel((next(iter(test_dataloader))[0])))

# loss_fun = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(elmodel.parameters(), lr=1e-3)

# # train_epoch_LM(elmodel, loss_fun, test_dataloader, optimizer)

# train_lstm_lm(TRAIN_FILE_PATH, language, "elmo.pth", NUM_EPOCHS, VALIDATION_FILE_PATH)

train_elmo_classification("elmo_f.pth", TRAIN_FILE_PATH, VALIDATION_FILE_PATH, language_object=language, num_epochs=NUM_EPOCHS)
