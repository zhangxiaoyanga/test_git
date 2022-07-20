from utils import *
from model import *
from config import *
from seqeval.metrics import f1_score, precision_score, recall_score
import os


def train():

    # 读取训练文件夹
    train_filename = os.path.join('data', 'train.txt')
    # 返回训练数据的文本和标签
    train_text, train_label = read_data(train_filename)

    # 验证集
    dev_filename = os.path.join('data', 'dev.txt')
    dev_text, dev_label = read_data(dev_filename)
    # print(train_filename)

    # 得到label2index, index2label
    label2index, index2label = build_label_2_index(train_label)

    # 数据迭代器
    train_data = Data(train_text, train_label, tokenizer, label2index, MAX_LEN)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=False)

    dev_data = Data(dev_text, dev_label, tokenizer, label2index, MAX_LEN)
    dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False)

    # 模型
    model = MyModel(len(label2index)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            batch_text, batch_label, batch_len = data
            # 将数据放到GPU上
            loss = model(batch_text.to(DEVICE), batch_label.to(DEVICE))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, BATCH: {batch_idx}, Training Loss:  {loss.item()}')
        # torch.save(model, MODEL_DIR + f'model_{epoch}.pth')

        model.eval()

        # 用来存放预测标签和真实标签
        all_pre = []
        all_tag = []

        for batch_text, batch_label, batch_len in dev_loader:

            # 因为是预测，所以在模型输入的地方，没有加入batch_label
            pre = model(batch_text.to(DEVICE))

            # 将pre从GPU上读下来，转成list
            pre = pre.cpu().numpy().tolist()
            batch_label = batch_label.cpu().numpy().tolist()

            # 还有一点要注意， from seqeval.metrics import f1_score
            # 在使用 f1_score的时候，所需要的标签应该是完整的，而不是经过填充过的
            # 所以我们需要将填充过的标签信息进行拆分怎么做呢？
            # 就需要将最开始没有填充过的文本长度记录下来，在__getitem__的返回量中增加一个长度量，那样我们就能知道文本真实长度
            # 然后就此进行切分，因为左边增加了一个开始符，需要去掉一个即可；右边按照长度来切分

            for p, t, l in zip(pre, batch_label, batch_len):
                p = p[1: l + 1]
                t = t[1: l + 1]

                pre = [index2label[j] for j in p]
                tag = [index2label[j] for j in t]
                all_pre.append(pre)
                all_tag.append(tag)
        f1_score_ = f1_score(all_pre, all_tag)
        p_score = precision_score(all_pre, all_tag)
        r_score = recall_score(all_pre, all_tag)
        # f1_score(batch_label_index, pre)
        print(f'p值={p_score}, r值={r_score}, f1={f1_score_}')
        # print(2*p_score*r_score/(p_score+r_score))


if __name__ == '__main__':
    train()
