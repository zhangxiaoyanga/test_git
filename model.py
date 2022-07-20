import torch.nn as nn
from config import *


class MyModel(nn.Module):
    def __init__(self, class_num):
        super(MyModel, self).__init__()
        self.class_num = class_num

        self.bert = BertModel.from_pretrained(BERT_PATH)

        self.lstm = nn.LSTM(768,
                            768 // 2,
                            bidirectional=True,
                            batch_first=True)

        self.linear = nn.Linear(768, class_num)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_text, batch_label=None):
        output = self.bert(batch_text)
        bert_out0, bert_out1 = output[0], output[1]
        output1, _ = self.lstm(bert_out0)
        pre = self.linear(output1)

        if batch_label is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), batch_label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)

