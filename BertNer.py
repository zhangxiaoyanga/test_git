import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel

from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score

from sklearn.metrics import f1_score as sklearn_f1_score


# 1、 读取数据
def read_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        all_data = f.read().split('\n')
        # print(all_data[0:10])
        # exit()

    # 用来存放所有的文本和标签
    all_text = []
    all_label = []

    # 用来存放一句话和标签
    text = []
    label = []

    for data in all_data:
        # 判断是否一句话结束
        if data == '':
            # 一句话结束了，需要将下面的text和label传入all——text中
            all_text.append(text)
            all_label.append(label)

            # 接下来就将存放一句话的text和label置为空,否则只会接受第一句代码
            text = []
            label = []

        else:
            t, l = data.split(' ')
            text.append(t)
            label.append(l)
    return all_text, all_label


# 2、 构建数据迭代器
class BertDataset(Dataset):
    # 在初始函数中，传入all_text和all_label,以及一些必要参数
    def __init__(self, all_text, all_label, label2index, tokenizer, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.label2index = label2index
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index]

        # 设置了最长长度，所以需要将label进行截断，不然仅仅对文本截断是不可以的，后面还是对不齐
        labels = self.all_label[index][:self.max_len]

        # 对文本进行编码
        text_index = self.tokenizer.encode(text,
                                           add_special_tokens=True,
                                           max_length=self.max_len + 2,  # 已经包含了左右字符，需要加个2
                                           padding='max_length',
                                           truncation=True,
                                           return_tensors='pt')
        # CLS  A  SEP   B  SEP
        # label_index = [self.label2index[label] for label in labels] 
        # 将label标签转为数字，让计算机看的懂 
        # 为了防止一个标签没有出现，从而报错，所以对上面的内容进行改写一下
        # label_index = [self.label2index.get(label, 1) for label in labels]

        # 到上面为止，虽然text和label都做好了标签，但是没有对齐啊
        # 在原来max_len的基础上要加2，是因为加了左右特殊标识符，其实不要也行吧
        # label没有对齐。需要进行处理
        label_index = [0] + [self.label2index.get(label, 1) for label in labels] + [0] + [0] * (
                self.max_len - len(text))

        # 转为tensor
        label_index = torch.tensor(label_index)
        # return torch.squeeze(text_index), label_index
        # 这里为啥要这样操作呢，是因为，如果不这样操作的话
        # batch_text_index的shape是（batch_size,1,max_len），但是我们并不要中间这维度1
        # 通常我们需要的是（batch_size,max_len）
        # 所以将其进行reshape操作
        return text_index.reshape(-1), label_index, len(labels)

    def __len__(self):
        return len(self.all_text)


# 3、在构建数据迭代器之前，我们要想机器了解内容，需要将内容进行编码
# 转换成机器能够看的懂的，将text和label进行编码
def bulid_label(train_label):
    # 在原始的label2index中加上“PAD”，“UNK”
    label_2_index = {'PAD': 0, 'UNK': 1}
    for labels in train_label:
        for label in labels:
            if label not in label_2_index:
                label_2_index[label] = len(label_2_index)
    return label_2_index, list(label_2_index)


# 4、 构建模型
class BertNerModel(nn.Module):
    def __init__(self, class_nums):
        super(BertNerModel, self).__init__()
        # self.bert = BertModel.from_pretrained(
        # 'D:\\ZhangYang\\实体识别练习\\BERT_MODEL\\chinese_wwm_pytorch\\pytorch_model.bin')
        config = BertConfig.from_pretrained(r'D:\\ZhangYang\\实体识别练习\BERT_MODEL\\bert-base-chinese\\bert_config.json')
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(768, 768//2, batch_first=True, bidirectional=True)
        # self.classifier = nn.Linear(config.hidden_size, class_nums)
        self.classifier = nn.Linear(config.hidden_size, class_nums)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_index, batch_label=None):
        output = self.bert(batch_index)
        # bert_out0.shape==[16, 32, 768]====字符级别, bert_out1.shape==[16, 768]====篇章级别
        bert_out0, bert_out1 = output[0], output[1]
        out, _ = self.lstm(bert_out0)
        # out.shape == torch.Size([64, 32, 768])
        pre = self.classifier(out)
        # pre.shape == torch.Size([64, 32, 30])
        if batch_label is not None:
            # loss = self.loss_fn(pre, batch_label)
            # 此时，pre.shape==[16, 32, 30]---[batch_size,seq_len,output_size], batch_label.shape==[16, 32]
            # nn.CrossEntropyLoss(),已经做了softmax，不能计算三维的，只能计算两维的。需要reshape
            # 因为我们需要计算每一个标签的概率（这里是三十个），所以需要将数据的维度进行改变
            # 因为要做三十个标签概率，将预测的维度变成：（class_num, batch_size*seq_len）；目标标签的维度变成：（-1）
            # 1)模型原始预测输出, shape(n_samples, n_class), dtype(torch.float)
            # 2)真实标签, shape = (n_samples), dtype(torch.long)

            # 作者是说，batch=16，seq=32，一共有16篇文章，每篇32个字，将其合成一个维度，即：16*32
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), batch_label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)


if __name__ == "__main__":
    # train_text, train_label = read_data(os.path.join('手写AI识别\data', 'train.txt'))
    train_text, train_label = read_data('D:\\ZhangYang\\实体识别练习\\手写AI识别\\data\\train.txt')
    # dev_text, dev_label = read_data(os.path.join('手写AI识别\data', 'dev.txt'))
    dev_text, dev_label = read_data('D:\\ZhangYang\\实体识别练习\\手写AI识别\\data\\dev.txt')
    # test_text, test_label = read_data(os.path.join('手写AI识别\data', 'test.txt'))
    test_text, test_label = read_data('D:\\ZhangYang\\实体识别练习\\手写AI识别\\data\\test.txt')

    label2index, index2label = bulid_label(train_label)

    # 定义一些超参数
    batch_size = 64
    epochs = 100
    max_len = 30
    lr = 1e-5

    # 分词器

    tokenizer = BertTokenizer.from_pretrained('D:\\ZhangYang\\实体识别练习\\BERT_MODEL\\roberta\\vocab.txt')

    # 训练数据迭代器
    train_dataset = BertDataset(train_text, train_label, label2index, tokenizer, max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # 验证数据迭代器
    dev_dataset = BertDataset(dev_text, dev_label, label2index, tokenizer, max_len)

    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # cuda调用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型
    model = BertNerModel(len(label2index)).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()
        for batch_text_index, batch_label_index, batch_len in train_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)

            loss = model(batch_text_index, batch_label_index)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            print(f'epoch:{epoch},loss:{loss.item():.4f}')

        model.eval()

        all_pre = []
        all_tag = []
        for batch_text_index, batch_label_index, batch_len in dev_dataloader:
            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            pre = model(batch_text_index)

            # 因为seq_f1_score只能识别B-I-等等，需要将index转为label

            # seq_f1_score(batch_label_index, pre),就会报错。
            # Found input variables without list of list.
            pre = pre.cpu().numpy().tolist()
            tag = batch_label_index.cpu().numpy().tolist()

            # pre = [[index2label[j] for j in i] for i in pre]
            # batch_label_index = [[index2label[j] for j in i] for i in batch_label_index]

            # 还有一点要注意
            # 我们在使用 seq_f1_score的时候，所需要的标签应该是完整的，而不是经过填充过的
            # 所以我们需要将填充过的标签信息进行拆分怎么做呢？
            # 就需要将最开始没有填充过的文本长度记录下来，在__getitem__的返回量中增加一个长度量
            # 然后就此进行切分，因为左边增加了一个开始符，需要去掉一个即可；右边按照长度来切分
            # 设置两个列表来存预测值和真实值

            for p, t, l in zip(pre, tag, batch_len):
                p = p[1: 1 + l]
                t = t[1: 1 + l]

                p = [index2label[j] for j in p]
                t = [index2label[j] for j in t]

                all_pre.append(p)
                all_tag.append(t)

        f1_score = seq_f1_score(all_pre, all_tag)
        # f1_score(batch_label_index, pre)
        print(f'f1={f1_score}')
