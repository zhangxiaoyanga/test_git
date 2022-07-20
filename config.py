import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
MAX_LEN = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BERT_PATH = r'D:\ZhangYang\实体识别练习\BERT_MODEL\roberta'

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
MODEL_DIR = 'model/'


