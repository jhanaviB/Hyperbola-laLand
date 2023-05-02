PATH_HYPO_L = 'data/HYPO-L.csv'
PATH_HYPO_XL = 'data/HYPO-XL.txt'
PATH_GLOVE_EMBED = 'data/glove.840B.300d.txt'
PATH_POS_PATTERNS = "data/pattern.tsv"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
PATH_BART_CHECKPOINT = "checkpoints/model.pickle"
PATH_BERT_CSV = "data/Bert.csv"
hyperparameters = {
    "learning_rate": 0.0001,
    "num_epochs": 2,  # set to very high number
    # Actual batch size will this x 8 (was 8 before but can cause OOM)
    "train_batch_size": 1,
    # Actual batch size will this x 8 (was 32 before but can cause OOM)
    "eval_batch_size": 1,
    "seed": 42,
    "patience": 3,  # early stopping
    "output_dir": "/content/",
}