from utils import *
from model import *
from config import *
import os


def predict():
    train_filename = os.path.join('data', 'train.txt')
    train_text, train_label = read_data(train_filename)

    test_filename = os.path.join('data', 'test.txt')
    test_text, _ = read_data(test_filename)
    text = test_text[1]

    print(text)

    inputs = tokenizer.encode(text,
                              return_tensors='pt')
    inputs = inputs.to(DEVICE)
    model = torch.load(MODEL_DIR + 'model_1.pth')
    y_pre = model(inputs).reshape(-1)  # 或者是y_pre[0]也行,因为y_pre是一个batch，需要进行reshape

    _, id2label = build_label_2_index(train_label)

    label = [id2label[l] for l in y_pre[1:-1]]
    print(text)
    print(label)


if __name__ == '__main__':
    predict()
