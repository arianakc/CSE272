import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from src.data_processor import read_ohsumed, read_queries, read_qrels_to_trec
from nltk.tokenize import word_tokenize
from torchtext.data import Field, LabelField, Example, Dataset, BucketIterator
import torch.optim as optim

import spacy
import json
spacy_en = spacy.load('en_core_web_sm')


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


class classifier(nn.Module):

    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        # Constructor
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)
        self.attn = nn.Linear((hidden_dim * 2) * 2 + 2 * hidden_dim, 2 * hidden_dim)
        self.v = nn.Linear(2 * hidden_dim, 1, bias=False)
        # dense layer
        self.fc = nn.Linear(hidden_dim * 10, output_dim)
        # activation function
        self.act = nn.Sigmoid()

    def forward(self, query_title, query_title_lengths, query_description, query_description_lengths, doc_text,
                doc_text_lengths):
        # text = [batch size,sent_length]
        query_title_embedded = self.embedding(query_title)
        query_description_embedded = self.embedding(query_description)
        doc_text_embedded = self.embedding(doc_text)

        # embedded = [batch size, sent_len, emb dim]
        # packed sequence
        packed_query_title_embedded = nn.utils.rnn.pack_padded_sequence(query_title_embedded, query_title_lengths.cpu(),
                                                                        batch_first=True,
                                                                        enforce_sorted=False)
        packed_query_description_embedded = nn.utils.rnn.pack_padded_sequence(query_description_embedded,
                                                                              query_description_lengths.cpu(),
                                                                              batch_first=True,
                                                                              enforce_sorted=False)
        packed_doc_text_embedded = nn.utils.rnn.pack_padded_sequence(doc_text_embedded,
                                                                     doc_text_lengths.cpu(),
                                                                     batch_first=True,
                                                                     enforce_sorted=False)

        _, (query_title_hidden, query_title_cell) = self.lstm(packed_query_title_embedded)
        _, (query_description_hidden, query_description_cell) = self.lstm(packed_query_description_embedded)
        _, (doc_text_hidden, doc_text_cell) = self.lstm(packed_doc_text_embedded)

        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        query_title_hidden = torch.cat((query_title_hidden[-2, :, :], query_title_hidden[-1, :, :]), dim=1)
        query_description_hidden = torch.cat((query_description_hidden[-2, :, :], query_description_hidden[-1, :, :]),
                                             dim=1)
        doc_text_hidden = torch.cat((doc_text_hidden[-2, :, :], doc_text_hidden[-1, :, :]), dim=1)
        query_hidden = torch.cat((query_title_hidden, query_description_hidden), dim=1)
        # hidden = [batch size, hid dim * num directions]
        energy = torch.tanh(self.attn(torch.cat((query_hidden, doc_text_hidden), dim=1)))
        # energy = [batch size, hid dim * 2]
        attention = self.v(energy)
        # attention = [batch size]
        dense_outputs = self.fc(torch.cat((query_hidden, attention * query_hidden, doc_text_hidden), dim=1))
        # Final activation function
        outputs = self.act(dense_outputs)
        return outputs


def create_document_dict(documents):
    doc_dict = {}
    for document in documents:
        doc_dict[document[".U"]] = document
    return doc_dict


def build_dataset(doc_dict, queries, qrels):
    examples = []
    n_positive = 0
    n_negative = 0
    for query in tqdm(queries, desc="create examples", total=len(queries)):
        qrel_doc_ids = [qrel["docno"] for qrel in qrels if qrel["qid"] == query["Number"]]
        for qrel_doc_id in qrel_doc_ids:
            document = doc_dict[qrel_doc_id]
            example = {"query_title": query["title"], "query_description": query["description"], "label": 1}
            if ".W" in document and ".M" in document:
                example["doc_text"] = document[".M"].lower() + " " + document[".T"].lower() + document[".W"].lower()
            elif ".M" in document and ".W" not in document:
                example["doc_text"] = document[".M"].lower() + " " + document[".T"].lower()
            elif ".M" not in document and ".W" in document:
                example["doc_text"] = "text", document[".T"].lower() + document[".W"].lower()
            elif ".M" not in document and ".W" not in document:
                example["doc_text"] = document[".T"].lower()
            examples.append(example)
            n_positive+=1
        for doc_id in doc_dict:
            if doc_id in qrel_doc_ids:
                continue
            else:
                document = doc_dict[doc_id]
                example = {"query_title": query["title"], "query_description": query["description"], "label": 0}
                if ".W" in document and ".M" in document:
                    example["doc_text"] = document[".M"].lower() + " " + document[".T"].lower() + document[".W"].lower()
                elif ".M" in document and ".W" not in document:
                    example["doc_text"] = document[".M"].lower() + " " + document[".T"].lower()
                elif ".M" not in document and ".W" in document:
                    example["doc_text"] = "text", document[".T"].lower() + document[".W"].lower()
                elif ".M" not in document and ".W" not in document:
                    example["doc_text"] = document[".T"].lower()
            examples.append(example)
            n_negative += 1
    print("total number of positive examples:", n_positive)
    print("total number of negative examples:", n_negative)
    print("total number of examples:", str(len(examples)))
    return examples


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion, epoch):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for batch in tqdm(iterator, desc="epoch-"+str(epoch), total=len(iterator)):
        # resets the gradients after every batch
        optimizer.zero_grad()
        # retrieve text and no. of words
        query_title, query_title_lengths = batch.query_title
        query_description, query_description_lengths = batch.query_description
        doc_text, doc_text_lengths = batch.doc_text

        # convert to 1D tensor
        predictions = model(query_title, query_title_lengths,
                            query_description, query_description_lengths,
                            doc_text, doc_text_lengths).squeeze()

        # compute the loss
        loss = criterion(predictions, batch.label)
        # compute the binary accuracy
        acc = binary_accuracy(predictions, batch.label)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def load_trained_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


def prepare_text_tensor(text):
    text_tokenized = tokenize_en(text)  # tokenize the sentence
    text_indexed = [TEXT.vocab.stoi[t] for t in text_tokenized]  # convert to integer sequence
    text_length = [len(text_indexed)]  # compute no. of words
    text_tensor = torch.LongTensor(text_indexed).to(device)  # convert to tensor
    text_tensor = text_tensor.unsqueeze(1).T  # reshape in form of batch,no. of words
    text_length_tensor = torch.LongTensor(text_length)  # convert to tensor
    return text_tensor, text_length_tensor


def predict(model, query_title, query_description, doc_text):
    query_title_tensor, query_title_length_tensor = prepare_text_tensor(query_title)
    query_description_tensor, query_description_length_tensor = prepare_text_tensor(query_description)
    doc_text_tensor, doc_text_length_tensor = prepare_text_tensor(doc_text)
    score = model(query_title_tensor, query_title_length_tensor,
                  query_description_tensor, query_description_length_tensor,
                  doc_text_tensor, doc_text_length_tensor)  # prediction
    return score.item()



def save_examples(dataset, savepath):
    with open(savepath, 'w') as f:
        # Save num. elements (not really need it)
        f.write(json.dumps(len(dataset.examples)))  # Write examples length
        f.write("\n")

        # Save elements
        for pair in dataset.examples:
            data = [pair.query_title, pair.query_description, pair.doc_text, pair.label]
            f.write(json.dumps(data))  # Write samples
            f.write("\n")


def load_examples(filename, fields):
    examples = []
    with open(filename, 'r') as f:
        # Read num. elements (not really need it)
        total = json.loads(f.readline())
        # Save elements
        for i in tqdm(range(total), desc="load examples"):
            line = f.readline()
            example = json.loads(line)
            example = Example().fromlist(example, fields)  # Create Example obj. (you can do it here or later)
            examples.append(example)
    return examples

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    build_new_data = False
    if build_new_data:
        feedback_documents = read_ohsumed("../data/ohsumed.87")
        queries = read_queries("../data/query.ohsu.1-63")
        qrels = read_qrels_to_trec("../data/qrels.ohsu.batch.87")
        doc_dict = create_document_dict(feedback_documents)
        examples = build_dataset(doc_dict, queries, qrels)
        TEXT = Field(tokenize=tokenize_en, batch_first=True, include_lengths=True)
        LABEL = LabelField(dtype=torch.float, batch_first=True)
        fields_examples = {"query_title": ("query_title", TEXT), "query_description": ("query_description", TEXT),
                           "doc_text": ("doc_text", TEXT), "label": ("label", LABEL)}
        torch_examples = []
        for example in tqdm(examples, desc="build_torch_examples", total=len(examples)):
            torch_examples.append(Example.fromdict(example, fields_examples))
        print("build_torch_dataset...")
        fields_dataset = [("query_title", TEXT), ("query_description", TEXT), ("doc_text", TEXT), ("label", LABEL)]
        train_data = Dataset(torch_examples, fields_dataset)
        save_examples(train_data, "../traindata.json")
        exit(0)
    else:
        TEXT = Field(tokenize=tokenize_en, batch_first=True, include_lengths=True)
        LABEL = LabelField(dtype=torch.float, batch_first=True)
        fields_dataset = [("query_title", TEXT), ("query_description", TEXT), ("doc_text", TEXT), ("label", LABEL)]
        train_data = Dataset(load_examples("../traindata.json", fields_dataset), fields_dataset)
    print("build_vocabulary...")
    TEXT.build_vocab(train_data, min_freq=1, vectors="glove.6B.300d")
    LABEL.build_vocab(train_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("build_iterator...")
    train_iterator, vaild_iterator = BucketIterator.splits(
        (train_data, train_data),
        batch_size=64,
        sort_key=lambda x: len(x.doc_text),
        sort_within_batch=False,
        device=device)

    size_of_vocab = len(TEXT.vocab)
    embedding_dim = 300
    num_hidden_nodes = 128
    num_layers = 2
    num_output_nodes = 1
    dropout = 0.2
    print("build model...")
    model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers, dropout=dropout)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("load pretrained embedding")
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    print(pretrained_embeddings.shape)
    print("build optimizer...")
    optimizer = optim.Adam(model.parameters())
    print("build criterion...")
    # weight = torch.tensor([2.0])
    weight = torch.tensor([5144.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    model = model.to(device)
    criterion = criterion.to(device)
    N_EPOCHS = 100
    best_train_loss = float('inf')
    print("start to train....")
    for epoch in range(N_EPOCHS):
        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, epoch+1)
        # save the best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(),
                       'saved_weights_epoch' + str(epoch+1) + '_trainloss' + str(train_loss) + '_trainacc' + str(
                           train_acc) + '.pt')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
