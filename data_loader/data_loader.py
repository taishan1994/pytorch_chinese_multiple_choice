import random
import sys

from tqdm import tqdm

sys.path.append("..")
import json
import jieba
import torch
from functools import reduce
from operator import concat
from torch.utils.data import DataLoader, Dataset

from utils.utils import sequence_padding


class NormalListDataset(Dataset):
    def __init__(self, file_path=None, data=None, tokenizer=None, shuffle=False, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path, tokenizer, shuffle=shuffle)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path, tokenizer, shuffle):
        return file_path


class NormalTokenizer:
    def __init__(self, words):
        self.words = words
        self.tmp = {"pad": 0, "unk": 1}
        self.word2id = {word: i + 2 for i, word in enumerate(words)}
        self.id2word = {i + 2: word for i, word in enumerate(words)}
        self.word2id.update(self.tmp)
        self.id2word.update(self.tmp)
        print(self.word2id)

    def convert_tokens_to_ids(self, tokens):
        return [self.word2id.get(word, 1) for word in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id2word[did] for did in ids]


# 加载数据集
class NormalDataset(NormalListDataset):
    @staticmethod
    def load_data(filename, tokenizer, shuffle):
        data = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            f = json.loads(f)
            if shuffle:
              random.shuffle(f)
            for i,d in tqdm(enumerate(f), total=len(f)):
                tmp = {}
                options = d['option_list']
                statement = d['statement']
                if "answer" in d:
                  labels = d['answer']
                else:
                  labels = []
                  tmp["id"] = d["id"] 
                option_input_ids = []
                for option in options.values():
                    tokens = jieba.lcut(option, cut_all=False)
                    option_token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    option_input_ids.append(option_token_ids)
                quesion_tokens = jieba.lcut(statement, cut_all=False)
                quesion_token_ids = tokenizer.convert_tokens_to_ids(quesion_tokens)
                quesion_token_ids = [quesion_token_ids for _ in range(4)]
                tmp["option_input_ids"] = option_input_ids
                tmp["quesion_token_ids"] = quesion_token_ids
                labels = [i for i in labels if i in ["A", "B", "C", "D"]]
                tmp["labels"] = labels
                # if i < 5:
                #   print(d)
                #   print(tmp)
                data.append(tmp)
        return data


class TestNormalCollate:
    def __init__(self, tag2id, device):
        self.tag2id = tag2id
        self.device = device

    def collate_fn(self, batch):
        batch_ids = []
        batch_option_token_ids = []
        batch_quesion_token_ids = []
        option_max_len = 0
        question_max_len = 0
        for i, d in enumerate(batch):
            option_input_ids = d["option_input_ids"]
            quesion_token_ids = d["quesion_token_ids"]
            did = d["id"]
            # token_ids = reduce(concat, token_ids)
            len1 = max([len(i) for i in option_input_ids])
            len2 = max([len(i) for i in quesion_token_ids])
            if len1 > option_max_len:
                option_max_len = len1
            if len2 > question_max_len:
                question_max_len = len2
            batch_option_token_ids.extend(option_input_ids)
            batch_quesion_token_ids.extend(quesion_token_ids)
            batch_ids.append(did)
        batch_option_token_ids = torch.tensor(sequence_padding(batch_option_token_ids, length=option_max_len),
                                              dtype=torch.long,
                                              device=self.device)
        batch_quesion_token_ids = torch.tensor(
            sequence_padding(batch_quesion_token_ids, length=question_max_len),
            dtype=torch.long,
            device=self.device)

        return batch_option_token_ids, batch_quesion_token_ids, batch_ids


class NormalCollate:
    def __init__(self, tag2id, device):
        self.tag2id = tag2id
        self.device = device

    def collate_fn(self, batch):
        batch_labels = []
        batch_option_token_ids = []
        batch_quesion_token_ids = []
        option_max_len = 0
        question_max_len = 0
        for i, d in enumerate(batch):
            option_input_ids = d["option_input_ids"]
            quesion_token_ids = d["quesion_token_ids"]
            tmp_labels = d["labels"]
            tmp_labels = [self.tag2id[label] for label in tmp_labels]
            labels = [0] * 4
            for label in tmp_labels:
                labels[label] = 1
            # token_ids = reduce(concat, token_ids)
            len1 = max([len(i) for i in option_input_ids])
            len2 = max([len(i) for i in quesion_token_ids])
            if len1 > option_max_len:
                option_max_len = len1
            if len2 > question_max_len:
                question_max_len = len2
            batch_option_token_ids.extend(option_input_ids)
            batch_quesion_token_ids.extend(quesion_token_ids)
            batch_labels.append(labels)
        batch_option_token_ids = torch.tensor(sequence_padding(batch_option_token_ids, length=option_max_len),
                                              dtype=torch.long,
                                              device=self.device)
        batch_quesion_token_ids = torch.tensor(
            sequence_padding(batch_quesion_token_ids, length=question_max_len),
            dtype=torch.long,
            device=self.device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=self.device)
        return batch_option_token_ids, batch_quesion_token_ids, batch_labels
