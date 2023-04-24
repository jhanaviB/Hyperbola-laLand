class BertDataset(Dataset):
            def __init__(self, tokenizer, max_length):
                super(BertDataset, self).__init__()
                self.train_csv = pd.read_csv(
                    '/content/drive/MyDrive/data/mover/Bert.csv', header=None).iloc[1:]
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


        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

        dataset = BertDataset(tokenizer, max_length=512)

        dataloader = DataLoader(dataset=dataset, batch_size=32)

        dataset.train_csv


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


        model = BERT()

        loss_fn = nn.BCEWithLogitsLoss()

        # Initialize Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for param in model.bert_model.parameters():
            param.requires_grad = False


        def finetune(epochs, dataloader, model, loss_fn, optimizer):
            model.to(device)
            model.train()
            for epoch in range(epochs):
                print(epoch)

                loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
                for batch, dl in loop:
                    ids = dl['ids']
                    token_type_ids = dl['token_type_ids']
                    mask = dl['mask']
                    label = dl['target']
                    label = label.unsqueeze(1)

                    optimizer.zero_grad()

                    output = model(
                        ids=ids.to(device),
                        mask=mask.to(device),
                        token_type_ids=token_type_ids.to(device))
                    label = label.type_as(output)

                    loss = loss_fn(output, label)
                    loss.backward()

                    optimizer.step()

                    pred = torch.where(output >= 0, torch.tensor(1), torch.tensor(0))

                    num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
                    num_samples = pred.shape[0]
                    accuracy = num_correct/num_samples

                    print(
                        f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

                    # Show progress while training
                    loop.set_description(f'Epoch={epoch}/{epochs}')
                    loop.set_postfix(loss=loss.item(), acc=accuracy)

            return model


        model = finetune(5, dataloader, model, loss_fn, optimizer)

        hypo_frame = pd.read_csv(
            '/content/drive/MyDrive/data/mover/HYPO.tsv', delimiter='\t')
        hypo_frame1 = pd.DataFrame(pd.concat([hypo_frame['HYPO'], hypo_frame['PARAPHRASES'],
                                  hypo_frame['MINIMAL UNITS CORPUS']], axis=0), columns=['SENTENCES'])
        hypo_frame1['LABELS'] = [1 if i < 709 else 0 for i in range(2127)]
        hypo_l = hypo_l.iloc[1:]
        hypo_frame1 = pd.concat([hypo_l['sentence'], hypo_l['label']], axis=1)
        hypo_frame1.to_csv('Bert.csv', index=False)
        print(hypo_frame1)


        def test_model(test_data, n):
            examples = test_data.original.sample(n)
            for sentence in examples:
                print("Original Sentence")
                print(sentence)
                print("Over Generation fine tuned")
                print(over_generate(sentence, model.to('cpu')))
                print()


        hypo_frame = pd.read_csv(
            '/content/drive/MyDrive/data/mover/HYPO.tsv', delimiter='\t')
        hypo_frame1 = pd.DataFrame(pd.concat([hypo_frame['HYPO'], hypo_frame['PARAPHRASES'],
                                  hypo_frame['MINIMAL UNITS CORPUS']], axis=0), columns=['SENTENCES'])
        hypo_frame1['LABELS'] = [1 if i < 709 else 0 for i in range(2127)]
        hypo_l = hypo_l.iloc[1:]
        hypo_frame1 = pd.concat([hypo_l['sentence'], hypo_l['label']], axis=1)
        hypo_frame1.to_csv('Bert.csv', index=False)
        print(hypo_frame1)