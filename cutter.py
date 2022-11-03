import os
from collections import Counter

import jieba
import json
from tqdm import tqdm

"""
该文件主要是进行分词获取词表
"""

def read_json_file(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as fp:
        data = json.load(fp)
    return data


def get_words_by_text(text):
    words = jieba.lcut(text, cut_all=False)
    return words


class SFKSProcess:
    def get_words_by_data(self, data):
        word_list = []
        for d in tqdm(data, ncols=100):
            option_list = d["option_list"]
            options = option_list.values()
            for option in options:
                words = get_words_by_text(option)
                word_list.extend(words)
            statement = d['statement']
            words = get_words_by_text(statement)
            word_list.extend(words)
        return word_list

    def sfks_process(self):
        train_data = read_json_file("data/sfks/raw_data/train.json")
        train_word_list = self.get_words_by_data(train_data)
        train_data = read_json_file("data/sfks/raw_data/test_input.json")
        test_word_list = self.get_words_by_data(train_data)
        word_list = train_word_list + test_word_list
        word_count = Counter(word_list)
        word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        word_count = {i: j for (i, j) in word_count}
        mid_data_path = 'data/sfks/mid_data/'
        if not os.path.exists(mid_data_path):
            os.mkdir(mid_data_path)

        with open(os.path.join(mid_data_path, "words.json"), "w", encoding="utf-8") as fp:
            fp.write(json.dumps(word_count, ensure_ascii=False))


if __name__ == '__main__':
    sfksprocess = SFKSProcess()
    sfksprocess.sfks_process()
