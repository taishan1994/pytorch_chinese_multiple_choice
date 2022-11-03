import json
import os
import logging
import random

import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config.config_parser import create_config
from data_loader.data_loader import NormalTokenizer, NormalDataset, NormalCollate, TestNormalCollate
from models.qa.qa import Model
from utils.utils import set_seed, set_logger, getstr

args = create_config("./config/sfks.config")
set_seed(args.getint("train", "seed"))
logger = logging.getLogger(__name__)

if args.getboolean("train", "use_tensorboard"):
    writer = SummaryWriter(log_dir='./tensorboard')


def cal_tp(preds, trues):
  tp = 0
  for i in range(len(preds)):
    tp += (preds[i] == trues[i])
  return tp

class MultipleChoiceTrainer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag, model, device):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        self.model = model
        self.device = device
        self.epochs = self.args.getint("train", "epochs")
        if train_loader is not None:
            self.t_total = len(self.train_loader) * self.epochs
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.getfloat("train", "learning_rate"))

    
    def save_model(self):
        output_dir = getstr(self.args.get("train", "output_dir"))
        output_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        logger.info('保存模型到 {}'.format(output_dir))
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))


    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = self.args.getfloat("train", "eval_steps")  # 每多少个step打印损失及进行验证
        best_acc = 0.
        for epoch in range(1, self.epochs + 1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data:
                    batch = batch.to(self.device)
                output = self.model(batch_data[0], batch_data[1], batch_data[2])
                loss = output["loss"]
                logits = output["logits"]
                # loss.backward(loss.clone().detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.getfloat("train", "max_grad_norm"))
                self.optimizer.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))

                global_step += 1
                if global_step % eval_steps == 0:
                  # self.predict(logits, batch_data[2])
                  accuracy = self.dev()
                  if best_acc < accuracy:
                    best_acc = accuracy
                    logger.info("【best acc】{}".format(best_acc))
                    self.save_model()
                if self.args.getboolean("train", "use_tensorboard"):
                    writer.add_scalar('data/loss', loss.item(), global_step)

    def dev(self):
      self.model.eval()
      tp_all = 0
      total = 0
      with torch.no_grad():
        for step, batch_data in enumerate(self.dev_loader):
            for batch in batch_data:
                batch = batch.to(self.device)
            output = self.model(batch_data[0], batch_data[1])
            logits = output["logits"]
            preds = np.where(logits.detach().cpu().numpy() > 0.5, 1., 0.).tolist()
            trues = batch_data[2].detach().cpu().numpy().tolist()
            total = total + len(trues)
            tp_all = tp_all + cal_tp(preds, trues)
      accuracy = tp_all / total
      return accuracy
    
    def test(self, checkpoint_path):
        logger.info('加载模型：{}'.format(checkpoint_path))
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')), strict=True)
        self.model.to(self.device)
        self.model.eval()
        res = []
        with torch.no_grad():
          for step, batch_data in enumerate(self.test_loader):
            for batch in batch_data[:-1]:
                batch = batch.to(self.device)
            output = self.model(batch_data[0], batch_data[1])
            ids = batch_data[2]
            logits = output["logits"]
            preds = np.where(logits.detach().cpu().numpy() > 0.5, 1., 0.).tolist()
            for did, pred in zip(ids, preds):
              res.append(did + "\t" + "".join([self.idx2tag[i] for i,tag in enumerate(pred) if tag == 1]))
        with open("preds.txt", "w", encoding="utf-8") as fp:
          fp.write("\n".join(res))
        

    def predict(self, preds, trues):
        preds = np.where(preds.detach().cpu().numpy() > 0.5, 1., 0.)
        for i in range(len(trues)):
          print("预测：", preds[i])
          print("正确：", trues[i])
          print("="*50)



if __name__ == '__main__':
    data_name = 'sfks'
    model_name = "lstm"

    if data_name == "sfks":
        set_logger(os.path.join(getstr(args.get("train", "log_dir")), '{}.log'.format(model_name)))
        with open("./data/sfks/mid_data/words.json", encoding="utf-8") as fp:
            words = json.load(fp).keys()
        print(len(words) + 2)
        tokenizer = NormalTokenizer(words)
        data = NormalDataset(file_path='./data/sfks/raw_data/train.json',
                                      tokenizer=tokenizer,
                                      shuffle=True)
        test_dataset = NormalDataset(file_path='./data/sfks/raw_data/test_input.json',
                                      tokenizer=tokenizer,
                                      shuffle=False)

        # pprint(train_dataset[0])
        train_ratio = 0.9
        train_num = int(len(data) * train_ratio)
        train_dataset = data[:train_num]
        dev_dataset = data[train_num:]
        labels = ["A", "B", "C", "D"]
        id2tag = {}
        tag2id = {}
        for i, label in enumerate(labels):
            id2tag[i] = label
            tag2id[label] = i
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        collate = NormalCollate(tag2id=tag2id, device=device)
        test_collate = TestNormalCollate(tag2id=tag2id, device=device)
        batch_size = args.getint("train", "batch_size")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate.collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collate.collate_fn)
        model = Model(args, gpu_list=[])
        model.to(device)

        multipleChoiceTrainer = MultipleChoiceTrainer(
            args,
            train_loader,
            dev_loader,
            test_loader,
            id2tag,
            model,
            device
        )

        multipleChoiceTrainer.train()

        checkpoint_path = getstr(args.get("train", "output_dir"))
        checkpoint_path = os.path.join(checkpoint_path, model_name)
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        multipleChoiceTrainer.test(checkpoint_path)