import sys
from pprint import pprint

sys.path.append("..")
import torch
import json
from torch.utils.data import DataLoader
from data_loader.data_loader import NormalTokenizer, NormalDataset, NormalCollate

with open("../data/sfks/mid_data/words.json", encoding="utf-8") as fp:
    words = json.load(fp).keys()
print(len(words) + 2)
tokenizer = NormalTokenizer(words)
train_dataset = NormalDataset(file_path='../data/sfks/raw_data/train.json',
                              tokenizer=tokenizer)
train_dataset = train_dataset[:10]
# pprint(train_dataset[0])

labels = ["A", "B", "C", "D"]
id2tag = {}
tag2id = {}
for i, label in enumerate(labels):
    id2tag[i] = label
    tag2id[label] = i
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
collate = NormalCollate(tag2id=tag2id, device=device)
batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

for i, batch in enumerate(train_dataloader):
    # print(batch[0].shape)
    # print(batch[1].shape)
    # print(batch[2].shape)
    print(batch[0])
    print(batch[1])
    print(batch[2])
    break