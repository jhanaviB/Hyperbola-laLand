from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd


class TrainDistilRoberta():
    def __init__(self) -> None:
        hypo_frame = pd.read_csv(
            '/content/drive/MyDrive/data/mover/HYPO.tsv', delimiter='\t')
        self.content_preservation_dataset = pd.DataFrame(
            columns=["Sentence_1", "Sentence_2", "Similar_Meaning"])
        content_preservation_dataset_index = 0
        for i in hypo_frame.index:
            self.content_preservation_dataset.loc[content_preservation_dataset_index] = [
                hypo_frame['HYPO'][i], hypo_frame['PARAPHRASES'][i], 1]
            content_preservation_dataset_index += 1
            self.content_preservation_dataset.loc[content_preservation_dataset_index] = [
                hypo_frame['HYPO'][i], hypo_frame['MINIMAL UNITS CORPUS'][i], 0]
            content_preservation_dataset_index += 1

        self.content_preservation_dataset.to_csv(
            "/content/drive/MyDrive/data/mover/content_preservation_dataset.csv")

    def process_inputExample(self, df):
        train_examples = []
        for i in df.index:
            train_examples.append(InputExample(
                texts=[df.Sentence_1[i], df.Sentence_2[i]], label=float(df.Similar_Meaning[i])))
        return train_examples

    def finetune(self):
        model = SentenceTransformer(
            'sentence-transformers/paraphrase-distilroberta-base-v1')

        train_examples = self.process_inputExample(
            self.content_preservation_dataset)
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model)
        model.fit(train_objectives=[
                  (train_dataloader, train_loss)], epochs=1, warmup_steps=100)

        # evaluator = evaluation.EmbeddingSimilarityEvaluator(
        #     sentences1, sentences2, scores)
