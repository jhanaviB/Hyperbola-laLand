import numpy as np
from config import MAX_TARGET_LENGTH, MAX_INPUT_LENGTH
# from torch.utils.data import Dataset
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from nltk.util import ngrams
import nltk


class DataProcessing():

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def matching(self, word_patterns, pos_patterns, tokens, patterns):
        """
        word_patterns = [['It', 'is'],['is', 'hardly'],['hardly', 'believable']]
        pos_patterns = [['PRP', 'VBZ'], ['VBZ', 'RB'], ['RB', 'JJ']]
        patterns = [['VBZ', 'RB'],["NN","IN"]]
        tokens = ['It','is','hardly','believable']

        Output: {
            1: [['is'],['hardly'],['believable']],
            2: [['hardly', 'believable'],['a', 'pretty']],
            3: [['bold', 'as', 'brass']],
            4: []
        }

        """
        candidates = []
        for i, pattern in enumerate(pos_patterns):
            if pattern in patterns:
                candidates.append(word_patterns[i])

        return candidates

    def mask(self, pdict_pattern, sentence):
        """
          sentence = It is hardly believable that such a pretty young lady is as bold as brass.\n
          pdict_pattern = { 1: [["NNP"],["NNS"]], 2: [["NNS,IN"],["NN","IN"]]}
          output = possible_candidates
        """
        possible_candidates = {}
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)

        for n_value, patterns in pdict_pattern.items():
            pos_patterns = []
            word_patterns = []
            # print(pos_tags)
            va = list(ngrams(pos_tags, n_value))
            for i in range(len(va)):
                vas = dict(va[i])
                pos_patterns.append(list(vas.values()))
                word_patterns.append(list(vas.keys()))

            possible_candidates[n_value] = self.matching(
                word_patterns, pos_patterns, tokens, patterns)

        return possible_candidates

    def mask_a_sentence(self, sentence, pos_n_gram_patterns, un_dict):

        res = list(zip(*un_dict))
        indices = np.argsort(res[0])[:3]
        n_grams = []
        for i in indices:
            n_grams.append(res[1][i])
        # print(n_grams)
        masked_sentences = []
        for i in n_grams:
            span = " ".join(i)
            masked_sentences.append(sentence.replace(span, "<mask>"))
        return masked_sentences

    def preprocess_examples(self, examples, tokenizer=None):
        # encode the documents
        inputs = examples['masked']
        outputs = examples['original']

        # inputs = [prefix + article for article in articles]
        model_inputs = tokenizer(
            inputs, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True)

        # encode the summaries
        labels = tokenizer(outputs, max_length=MAX_TARGET_LENGTH,
                           padding="max_length", truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label !=
                              0 else label for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

    def split_dataset(self, tokenizer):
        # Split the data into train, validation, and test sets
        train_data = pd.DataFrame.from_dict(
            self.dataset[:int(len(self.dataset)*0.6)])
        val_data = pd.DataFrame.from_dict(
            self.dataset[int(len(self.dataset)*0.6):int(len(self.dataset)*0.8)])
        test_data = pd.DataFrame.from_dict(
            self.dataset[int(len(self.dataset)*0.8):])

        # Convert the data into datasets
        train_ds = Dataset.from_pandas(train_data)
        val_ds = Dataset.from_pandas(val_data)
        test_ds = Dataset.from_pandas(test_data)

        encoded_train_ds = train_ds.map(
            self.preprocess_examples, batched=True,
            remove_columns=train_ds.column_names, fn_kwargs={"tokenizer": tokenizer})
        encoded_val_ds = val_ds.map(
            self.preprocess_examples, batched=True,
            remove_columns=val_ds.column_names, fn_kwargs={"tokenizer": tokenizer})
        encoded_test_ds = test_ds.map(
            self.preprocess_examples, batched=True,
            remove_columns=test_ds.column_names, fn_kwargs={"tokenizer": tokenizer})

        return encoded_train_ds, encoded_val_ds, encoded_test_ds

    def create_dataloaders(self, train_batch_size=8, eval_batch_size=32, tokenizer=None):

        # return train_dataloader, val_dataloader
        encoded_train_ds, encoded_val_ds, _ = self.split_dataset(tokenizer)
        train_dataloader = DataLoader(
            encoded_train_ds, shuffle=True, batch_size=train_batch_size)
        val_dataloader = DataLoader(
            encoded_val_ds, shuffle=False, batch_size=eval_batch_size)

        return train_dataloader, val_dataloader
