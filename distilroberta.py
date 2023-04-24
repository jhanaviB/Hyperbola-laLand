content_preservation_dataset.to_csv(
            "/content/drive/MyDrive/data/mover/content_preservation_dataset.csv")


def process_inputExample(df):
    train_examples = []
    for i in df.index:
        train_examples.append(InputExample(
            texts=[df.Sentence_1[i], df.Sentence_2[i]], label=float(df.Similar_Meaning[i])))
    return train_examples


model = SentenceTransformer(
    'sentence-transformers/paraphrase-distilroberta-base-v1')

train_examples = process_inputExample(content_preservation_dataset)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[
            (train_dataloader, train_loss)], epochs=1, warmup_steps=100)

sentences1 = ['We all cracked up at the joke.',
                'Every flavor is dynamite.', 'She is sinking in a sea of misery.']
sentences2 = ['We all laughed at the joke.', 'Love you so much.',
                'Pirates where sinking in a sea of cold water.']

scores = [1, 0, 0]

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1, sentences2, scores)
embeddings = model.encode([sentences1[0], sentences2[0]])
np.dot(embeddings[0], embeddings[1]) / \
    (np.linalg.norm(embeddings[0])*np.linalg.norm(embeddings[1]))