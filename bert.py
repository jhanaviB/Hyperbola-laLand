from torch.utils.data import Dataset
import pandas as pd
from config import PATH_BERT_CSV, MAX_INPUT_LENGTH
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import transformers
from bert_dataset import BertDataset
from tqdm.notebook import tqdm


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained(
            "bert-base-uncased")
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask,
                                token_type_ids=token_type_ids, return_dict=False)
        out = self.out(o2)

        return out


class TrainBert():
    def __init__(self, device='cpu', tokenizer=None) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=32)
        # self._load_dataset()
        self.dataset = BertDataset(self.tokenizer, max_length=MAX_INPUT_LENGTH)

    def _load_dataset(self):
        hypo_frame = pd.read_csv(
            '/content/drive/MyDrive/data/mover/HYPO.tsv', delimiter='\t')
        hypo_frame1 = pd.DataFrame(pd.concat([hypo_frame['HYPO'], hypo_frame['PARAPHRASES'],
                                              hypo_frame['MINIMAL UNITS CORPUS']], axis=0), columns=['SENTENCES'])
        hypo_frame1['LABELS'] = [1 if i < 709 else 0 for i in range(2127)]
        hypo_l = hypo_l.iloc[1:]
        hypo_frame1 = pd.concat([hypo_l['sentence'], hypo_l['label']], axis=1)
        hypo_frame1.to_csv('Bert.csv', index=False)

    def finetune(self, epochs):
        model = BERT()

        loss_fn = nn.BCEWithLogitsLoss()

        # Initialize Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for param in model.bert_model.parameters():
            param.requires_grad = False

        model.to(self.device)
        model.train()
        for epoch in range(epochs):
            print(epoch)

            loop = tqdm(enumerate(self.dataloader),
                        leave=False, total=len(self.dataloader))
            for batch, dl in loop:
                ids = dl['ids']
                token_type_ids = dl['token_type_ids']
                mask = dl['mask']
                label = dl['target']
                label = label.unsqueeze(1)

                optimizer.zero_grad()

                output = model(
                    ids=ids.to(self.device),
                    mask=mask.to(self.device),
                    token_type_ids=token_type_ids.to(self.device))
                label = label.type_as(output)

                loss = loss_fn(output, label)
                loss.backward()

                optimizer.step()

                pred = torch.where(
                    output >= 0, torch.tensor(1), torch.tensor(0))

                num_correct = sum(1 for a, b in zip(
                    pred, label) if a[0] == b[0])
                num_samples = pred.shape[0]
                accuracy = num_correct/num_samples

                print(
                    f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

                # Show progress while training
                loop.set_description(f'Epoch={epoch}/{epochs}')
                loop.set_postfix(loss=loss.item(), acc=accuracy)

        return model


# def test_model(test_data, n):
#     examples = test_data.original.sample(n)
#     for sentence in examples:
#         print("Original Sentence")
#         print(sentence)
#         print("Over Generation fine tuned")
#         print(over_generate(sentence, model.to('cpu')))
#         print()
