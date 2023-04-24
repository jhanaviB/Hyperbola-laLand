import torch
from data_processing import DataProcessing
from config import hyperparameters, MAX_INPUT_LENGTH, PATH_BART_CHECKPOINT
from transformers import BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np


class TrainBart():
    def __init__(self, optimizer, tokenizer, dataset, glove_embed, device='cpu') -> None:
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-base')
        self.data_processing = DataProcessing(dataset=dataset)
        self.glove_embeddings = glove_embed

    # Define a function to get the GloVe embedding of a word

    def get_glove_embedding(self, word):
        if word in self.glove_embeddings:
            return self.glove_embeddings[word]
        else:
            return torch.zeros(300)

    def unexpected_score_Un(self, sentence, n_gram_candidates):
        # word emebedding for one sentence
        unexpectedness_for_all_ngrams = []
        # Unexpected score Un
        for n_gram in n_gram_candidates:

            cos_sim = []

            text_to_ignore = " ".join(n_gram)
            new_sentence = sentence.replace(text_to_ignore, "", 1)
            tokenized_sentence = nltk.word_tokenize(new_sentence)

            for i in range(len(n_gram)):
                for j in range(len(tokenized_sentence)):
                    ge_ith_gram = self.get_glove_embedding(
                        n_gram[i]).reshape(1, -1)
                    ge_jth_token = self.get_glove_embedding(
                        tokenized_sentence[j]).reshape(1, -1)
                    cos_sim.append(cosine_similarity(
                        ge_ith_gram, ge_jth_token))

            unexpectedness_for_all_ngrams.append(np.mean(cos_sim))

        return zip(unexpectedness_for_all_ngrams, n_gram_candidates)

    def over_generate(self, sentence, pos_n_gram_patterns):
        possible_candidates = self.data_processing.mask(
            pos_n_gram_patterns, sentence)
        un_dict = []
        for i in range(len(possible_candidates)):
            un_dict += list(self.unexpected_score_Un(sentence,
                            possible_candidates[i+1]))

        masked_sentences = self.data_processing.mask_a_sentence(
            sentence, pos_n_gram_patterns, un_dict)
        print(masked_sentences)
        input_ids = self.tokenizer(
            masked_sentences, max_length=MAX_INPUT_LENGTH, padding="max_length")["input_ids"]
        logits = self.bart_model.generate(
            torch.tensor(input_ids).to(self.device))
        return self.tokenizer.batch_decode(logits, skip_special_tokens=True)

    def train(self):

        # Instantiate optimizer
        optimizer = self.optimizer(self.bart_model.parameters(),
                                   lr=hyperparameters["learning_rate"])

        for epoch in range(hyperparameters["num_epochs"]):
            # Define the model and tokenizer

            # Set the model to train mode
            self.bart_model.to(self.device)
            self.bart_model.train()

            # Prepare everything
            train_dataloader, val_dataloader = self.data_processing.create_dataloaders(
                train_batch_size=hyperparameters["train_batch_size"],
                eval_batch_size=hyperparameters["eval_batch_size"],
                tokenizer=self.tokenizer)

            # Loop through the training dataloader
            for step, batch in enumerate(train_dataloader):
                # Move the batch to the device

                # batch = {k: torch.tensor(v).to("cuda") for k, v in batch.items()}
                # Clear the gradients
                optimizer.zero_grad()

                # Get the input and target sequences
                input_ids = torch.tensor(
                    [[*x] for x in batch['input_ids']]).to(self.device)
                attention_mask = torch.tensor(
                    [[*x] for x in batch['attention_mask']]).to(self.device)
                target_ids = torch.tensor(
                    [[*x] for x in batch['labels']]).to(self.device)

                # Generate the outputs
                outputs = self.bart_model(input_ids=input_ids.T, attention_mask=attention_mask.T,
                                          labels=target_ids.T)

                # Compute the loss
                loss = outputs.loss

                # Compute the gradients
                loss.backward()

                # Clip the gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.bart_model.parameters(), max_norm=1.0)

                # Update the parameters
                optimizer.step()

                optimizer.zero_grad()

                # Print the loss every 10 steps
                if step % 20 == 0:
                    # logging.info(f"Step [{step}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
                    print(
                        f'Step [{step}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

                    # Evaluate at the end of the epoch (distributed evaluation as we have 8 TPU cores)
            self.bart_model.eval()
            validation_losses = []
            for batch in val_dataloader:
                input_ids = torch.tensor(
                    [[*x] for x in batch['input_ids']]).to(self.device)
                attention_mask = torch.tensor(
                    [[*x] for x in batch['attention_mask']]).to(self.device)
                target_ids = torch.tensor(
                    [[*x] for x in batch['labels']]).to(self.device)
                with torch.no_grad():
                    outputs = self.bart_model(input_ids=input_ids.T, attention_mask=attention_mask.T,
                                              labels=target_ids.T)
                loss = outputs.loss
                # We gather the loss from the 8 TPU cores to have them all.
                validation_losses.append(loss)

            # Compute average validation loss
            val_loss = torch.stack(validation_losses).sum(
            ).item() / len(validation_losses)
            # Use accelerator.print to print only on the main process.
            print(f"epoch {epoch}: validation loss:", val_loss)
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
                continue
            else:
                epochs_no_improve += 1
                # Check early stopping condition
                if epochs_no_improve == hyperparameters["patience"]:
                    print("Early stopping!")
                    break

    def load_prev_checkpoint(self, path=None):
        self.bart_model.load_state_dict(torch.load(
            PATH_BART_CHECKPOINT, map_location=torch.device(self.device)))
        self.bart_model.eval()

    def save_model(self):
        torch.save(self.bart_model.state_dict(), PATH_BART_CHECKPOINT)
