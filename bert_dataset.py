from torch.utils.data import Dataset
import pandas as pd
from config import PATH_BERT_CSV
import torch


class BertDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        super(BertDataset, self).__init__()
        self.train_csv = pd.read_csv(PATH_BERT_CSV, header=None).iloc[1:]
        self.tokenizer = tokenizer
        self.target = list(self.train_csv.iloc[1:, 1].astype('int32'))
        self.max_length = max_length

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, index):

        text1 = self.train_csv.iloc[index, 0]

        inputs = self.tokenizer.encode_plus(
            text1,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids),
            'mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'target': torch.tensor(self.target[index])
        }
