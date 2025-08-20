import pandas as pd
import json
import os
import unicodedata
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import time
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# =======================
# # Config
# =======================

MAX_LEN = 20
BATCH_SIZE = 32
EPOCHS = 4
DATA_FILE = 'uk_names_dataset.csv'
DATA_DIR = 'data_dir'
MODELS_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'name_ner_bert')
ARTIFACTS_DIR = os.path.join('name_ner_bert_artifacts')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =======================
# # Data Processing Utils
# =======================

def normalize_text(text):
    if not isinstance(text, str):
        return ''
    return unicodedata.normalize('NFKC', text).strip().upper()

def load_dataset_and_tag(file_path):
    print(f"\n=> Loading data from {file_path}...")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, file_path), low_memory=False)
    except FileNotFoundError:
        print(f"Error: Dataset file '{os.path.join(DATA_DIR, file_path)}' not found.")
        print("Please run the dataset generation script first and place the file in the 'data' directory.")
        exit()

    all_sentences, all_labels = [], []
    
    for _, row in df.iterrows():
        input_tokens = [normalize_text(t) for t in str(row['full_name_supplied']).replace(",", " ").split() if normalize_text(t)]
        
        labels_map = {
            'TITLE': normalize_text(row['Title']).split(),
            'FNAME': normalize_text(row['First name']).split(),
            'SNAME': normalize_text(row['Second name']).split(),
            'SURNAME': normalize_text(row['Surname']).split(),
            'SUFFIX': normalize_text(row['Suffix']).split()
        }

        bio_tags = ['O'] * len(input_tokens)
        processed_indices = set()

        for tag_type, part_tokens in labels_map.items():
            if not part_tokens:
                continue

            for i in range(len(input_tokens) - len(part_tokens) + 1):
                if tuple(input_tokens[i:i + len(part_tokens)]) == tuple(part_tokens):
                    bio_tags[i] = f'B-{tag_type}'
                    processed_indices.add(i)
                    for j in range(1, len(part_tokens)):
                        if i + j < len(bio_tags):
                            bio_tags[i+j] = f'I-{tag_type}'
                            processed_indices.add(i+j)
                    break
        
        if len(input_tokens) and len(input_tokens) == len(bio_tags):
            all_sentences.append(input_tokens)
            all_labels.append(bio_tags)

    print(f"\n=> Loaded and processed {len(all_sentences)} sequences.")
    return all_sentences, all_labels

# =======================
# # Main Processing
# =======================

if __name__ == "__main__":
    print("Starting BERT fine-tuning on structured dataset...")
    start_time = time.time()
    
    sentences, labels = load_dataset_and_tag(DATA_FILE)

    if not sentences:
        print("No data loaded. Exiting...")
        exit()

    # Get unique tags
    unique_tags = sorted(list(set(tag for sublist in labels for tag in sublist)))
    tag2idx = {tag: i for i, tag in enumerate(unique_tags)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    
    json.dump(tag2idx, open(os.path.join(ARTIFACTS_DIR, "tag2idx.json"), "w"))
    json.dump(idx2tag, open(os.path.join(ARTIFACTS_DIR, "idx2tag.json"), "w"))
    print(f"Tag classes: {len(tag2idx)}")

    # Load pre-trained tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize and align labels
    def tokenize_and_align_labels(sentences, labels):
        tokenized_inputs = []
        labels_aligned = []
        
        for words, word_labels in zip(sentences, labels):
            tokenized_sentence = []
            label_sequence = []
            
            for word, label in zip(words, word_labels):
                word_tokens = tokenizer.tokenize(word)
                tokenized_sentence.extend(word_tokens)
                label_sequence.extend([tag2idx[label]] * len(word_tokens))
                
            tokenized_inputs.append(tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_sentence + ['[SEP]']))
            
            # Pad or truncate
            tokenized_inputs[-1] = tokenized_inputs[-1][:MAX_LEN]
            label_sequence = label_sequence[:MAX_LEN - 2]
            
            # Pad
            tokenized_inputs[-1] = tokenized_inputs[-1] + [0] * (MAX_LEN - len(tokenized_inputs[-1]))
            label_sequence = label_sequence + [-100] * (MAX_LEN - len(label_sequence))
            
            labels_aligned.append(label_sequence)

        return torch.tensor(tokenized_inputs), torch.tensor(labels_aligned)

    print("\n=> Tokenizing and aligning labels...")
    input_ids, labels_aligned = tokenize_and_align_labels(sentences, labels)

    # Split data
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        input_ids, labels_aligned, test_size=0.1, random_state=42
    )

    # Create DataLoader
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_dataset = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE)

    # Model
    print("\n=> Building DistilBERT model...")
    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(tag2idx),
        id2label=idx2tag,
        label2id=tag2idx
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    print("\n=> Starting training for {} epochs...".format(EPOCHS))
    
    for epoch in range(EPOCHS):
        print("\n======== Epoch {} / {} ========".format(epoch + 1, EPOCHS))
        
        # Training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            
            outputs = model(b_input_ids, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(" Average training loss: {:.4f}".format(avg_train_loss))

        # Validation
        model.eval()
        eval_loss = 0
        
        for batch in tqdm(val_dataloader, desc="Validating"):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                outputs = model(b_input_ids, labels=b_labels)
            eval_loss += outputs.loss.item()
        
        avg_val_loss = eval_loss / len(val_dataloader)
        print(" Validation Loss: {:.4f}".format(avg_val_loss))

    # Save Model
    print("\n=> Saving final model...")
    model_save_path = os.path.join(MODELS_DIR, "name_ner_bert")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to: {model_save_path}")
    print("\nTRAINING COMPLETE!")