from util import *
from constant import *
from model import BiLSTM_CRF
import torch.optim as optim

# 准备训练数据
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

test_data=[("the Wall Street Journal reported on the outbreak in New York".split(),
           "B I I I O O O O O B I".split())]


word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for sentence, tags in test_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)


with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print("1:",model(precheck_sent))


for epoch in range(
        300):
    for sentence, tags in training_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    precheck_sent = prepare_sequence(test_data[0][0], word_to_ix)
    print("2:",model(precheck_sent))