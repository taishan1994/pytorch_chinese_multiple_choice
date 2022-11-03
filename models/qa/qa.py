import sys

sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder.LSTMEncoder import LSTMEncoder
from models.layer.Attention import Attention


class Model(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Model, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.vocab_size = config.getint("model", "vocab_size")
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.context_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.question_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        self.rank_module = nn.Linear(self.hidden_size * 2, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,
                option_token_ids,
                question_token_ids,
                labels=None
                ):
        batch = question_token_ids.size()[0] // 4
        option = 4

        context = self.embedding(option_token_ids)
        question = self.embedding(question_token_ids)

        _, context = self.context_encoder(context)
        _, question = self.question_encoder(question)

        c, q, a = self.attention(context, question)

        y = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)

        y = y.view(batch * option, -1)
        y = self.rank_module(y)

        y = y.view(batch, option)

        if labels is not None:
            loss = self.criterion(y, labels)
            return {"loss": loss, "logits": y}

        return {"logits": y}
