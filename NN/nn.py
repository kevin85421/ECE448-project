import pandas as pd

# preprocessing
import nltk
from nltk.corpus import stopwords
import re
import string

# NN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# doc2vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# split train_set & test_set
from sklearn.model_selection import train_test_split


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.model = torch.nn.Sequential(torch.nn.Linear(in_size, 32), torch.nn.ReLU(), torch.nn.Linear(32, out_size))
        # self.model = torch.nn.Sequential(torch.nn.Linear(in_size, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, out_size))
        self.optimizer = optim.SGD(self.model.parameters(), lr= self.lrate)

    def forward(self, x):
        return self.model(x) # Evaluate Fw(x)

    def step(self, x, y):
        self.optimizer.zero_grad()
        fw_x = self.forward(x)
        loss = self.loss_fn(fw_x, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    # Step1: Create NN
    lrate = 0.05
    loss_fn = torch.nn.CrossEntropyLoss()
    in_size = len(train_set[0])
    out_size = 2 
    nn = NeuralNet(lrate, loss_fn, in_size, out_size)

    # Step2: Normalize train_set
    normalize_train_set = (train_set - train_set.mean()) / train_set.std()

    # Step3: Train the NN
    losses = []

    for iteration in range(n_iter):
        batch_start = (iteration * batch_size) % len(train_set)
        batch_end = ((iteration + 1) * batch_size) % len(train_set)
        loss = nn.step(normalize_train_set[batch_start:batch_end], train_labels[batch_start:batch_end])
        losses.append(loss)

    # Step4: Normalize dev_set
    normalize_dev_set = (dev_set - train_set.mean()) / train_set.std()

    # Step5: Predict dev_set
    outputs = nn(normalize_dev_set)
    yhats_tensor = torch.argmax(outputs, dim=1)
    return losses, yhats_tensor, nn

def compute_accuracies(predicted_labels, dev_labels):
    acc = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == dev_labels[i]:
            acc += 1
    return (acc * 100) / len(dev_labels)


# Load the dataset with Pandas
dfTrue = pd.read_csv("../data/True.csv", sep=',',  encoding='utf8')
dfFake = pd.read_csv("../data/Fake.csv", sep=',',  encoding='utf8')
dfTrue['label'] = 1
dfFake['label'] = 0
df = pd.concat([dfFake, dfTrue], ignore_index=True)

# Preprocessing
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Train doc2Vec model
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
model = Doc2Vec(documents, vector_size=1024, window=2, min_count=1, workers=4)
df['vector'] = df['clean_text'].apply(lambda x: model.infer_vector(x.split()))

# split dataset to train_set and test_set
train_set, test_set, train_label, test_label = train_test_split(df['vector'], df['label'], test_size=0.3, shuffle=True)
train_tensor = torch.tensor(train_set.tolist())
train_label_tensor = torch.tensor(train_label.tolist())
test_tensor = torch.tensor(test_set.tolist()) 
test_label_tensor = torch.tensor(test_label.tolist())


_, predicted_labels, net = fit(train_tensor, train_label_tensor, test_tensor, 5000)
acc = compute_accuracies(predicted_labels, test_label_tensor)
print("accuracy: " + str(acc))

