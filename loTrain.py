import pandas as pd
import json
import os
import unicodedata
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import time
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import numpy as np

# =======================
# Enhanced Config
# =======================

MAX_LEN = 64  # Increased for longer names
BATCH_SIZE = 16  # Reduced for better gradients
EPOCHS = 8  # More epochs for better convergence
LEARNING_RATE = 2e-5  # Lower learning rate
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DATA_FILE = 'uk_names_dataset.csv'
DATA_DIR = 'data_dir'
MODELS_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'name_ner_bert')
ARTIFACTS_DIR = os.path.join('name_ner_bert_artifacts')

# Local pre-trained model path
LOCAL_BERT_PATH = "BERT"  # Path to your local BERT model directory

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =======================
# Enhanced Data Processing Utils
# =======================

def normalize_text(text):
    """Enhanced text normalization"""
    if not isinstance(text, str) or pd.isna(text):
        return ''
    # Handle various Unicode normalizations and clean up
    text = unicodedata.normalize('NFKC', str(text))
    # Remove extra whitespace and convert to upper
    text = re.sub(r'\s+', ' ', text.strip()).upper()
    return text

def augment_name_data(sentences, labels, augment_factor=2):
    """Data augmentation for name variations"""
    augmented_sentences = []
    augmented_labels = []
    
    for sent, lab in zip(sentences, labels):
        # Original
        augmented_sentences.append(sent)
        augmented_labels.append(lab)
        
        # Add variations
        for _ in range(augment_factor):
            # Random case changes
            new_sent = []
            for token in sent:
                if np.random.random() > 0.7:  # 30% chance to change case
                    if np.random.random() > 0.5:
                        new_sent.append(token.lower())
                    else:
                        new_sent.append(token.title())
                else:
                    new_sent.append(token)
            
            augmented_sentences.append(new_sent)
            augmented_labels.append(lab)
    
    return augmented_sentences, augmented_labels

def load_dataset_and_tag(file_path):
    print(f"\n=> Loading data from {file_path}...")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, file_path), low_memory=False)
    except FileNotFoundError:
        print(f"Error: Dataset file '{os.path.join(DATA_DIR, file_path)}' not found.")
        exit()

    all_sentences, all_labels = [], []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Clean input tokens
            input_text = str(row['full_name_supplied']).replace(",", " ")
            input_tokens = [normalize_text(t) for t in input_text.split() if normalize_text(t)]
            
            if not input_tokens or len(input_tokens) > MAX_LEN - 2:  # Account for [CLS] and [SEP]
                skipped_count += 1
                continue
            
            # Create labels mapping with better handling
            labels_map = {}
            for col, tag in [('Title', 'TITLE'), ('First name', 'FNAME'), 
                           ('Second name', 'SNAME'), ('Surname', 'SURNAME'), 
                           ('Suffix', 'SUFFIX')]:
                if col in row and pd.notna(row[col]):
                    tokens = [normalize_text(t) for t in str(row[col]).split() if normalize_text(t)]
                    if tokens:
                        labels_map[tag] = tokens

            bio_tags = ['O'] * len(input_tokens)
            
            # More robust token matching
            for tag_type, part_tokens in labels_map.items():
                # Try exact match first
                for i in range(len(input_tokens) - len(part_tokens) + 1):
                    if input_tokens[i:i + len(part_tokens)] == part_tokens:
                        bio_tags[i] = f'B-{tag_type}'
                        for j in range(1, len(part_tokens)):
                            if i + j < len(bio_tags):
                                bio_tags[i + j] = f'I-{tag_type}'
                        break
            
            all_sentences.append(input_tokens)
            all_labels.append(bio_tags)
            
        except Exception as e:
            skipped_count += 1
            continue

    print(f"\n=> Loaded {len(all_sentences)} sequences, skipped {skipped_count}")
    return all_sentences, all_labels

def tokenize_and_align_labels(sentences, labels, tokenizer, tag2idx, max_len):
    """Improved tokenization with proper label alignment"""
    tokenized_inputs = []
    labels_aligned = []
    attention_masks = []
    
    for words, word_labels in zip(sentences, labels):
        # Initialize with [CLS]
        tokens = ['[CLS]']
        aligned_labels = [-100]  # Ignore [CLS] in loss calculation
        
        # Process each word
        for word, label in zip(words, word_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:  # Skip if tokenization fails
                continue
                
            tokens.extend(word_tokens)
            # First subtoken gets the label, rest get -100
            aligned_labels.append(tag2idx[label])
            aligned_labels.extend([-100] * (len(word_tokens) - 1))
        
        # Add [SEP]
        tokens.append('[SEP]')
        aligned_labels.append(-100)
        
        # Convert to IDs and handle length
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Truncate if too long
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            aligned_labels = aligned_labels[:max_len]
            aligned_labels[-1] = -100  # Ensure last token is ignored
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = max_len - len(input_ids)
        input_ids.extend([tokenizer.pad_token_id] * padding_length)
        aligned_labels.extend([-100] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        tokenized_inputs.append(input_ids)
        labels_aligned.append(aligned_labels)
        attention_masks.append(attention_mask)

    return (torch.tensor(tokenized_inputs, dtype=torch.long), 
            torch.tensor(labels_aligned, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long))

def compute_metrics(predictions, true_labels, idx2tag):
    """Compute detailed metrics"""
    pred_flat = predictions.flatten()
    labels_flat = true_labels.flatten()
    
    # Filter out -100 labels
    mask = labels_flat != -100
    pred_flat = pred_flat[mask]
    labels_flat = labels_flat[mask]
    
    # Convert to label names
    pred_labels = [idx2tag[p] for p in pred_flat]
    true_labels_named = [idx2tag[l] for l in labels_flat]
    
    return classification_report(true_labels_named, pred_labels, output_dict=True)

# =======================
# Enhanced Training Loop
# =======================

def verify_local_model(model_path):
    """Verify that all required model files exist"""
    required_files = [
        'config.json',
        'model.safetensors',  # or 'pytorch_model.bin'
        'tokenizer_config.json',
        'tokenizer.json',
        'vocab.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        # Check for alternative model file names
        if 'model.safetensors' in missing_files and os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
            missing_files.remove('model.safetensors')
    
    return missing_files

def train_model():
    print("Starting Enhanced BERT fine-tuning with LOCAL pre-trained model...")
    start_time = time.time()
    
    # Verify local model files
    print(f"\n=> Checking local BERT model at: {LOCAL_BERT_PATH}")
    missing_files = verify_local_model(LOCAL_BERT_PATH)
    
    if missing_files:
        print(f"❌ Missing required files in {LOCAL_BERT_PATH}:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\nRequired files structure:")
        print(f"   {LOCAL_BERT_PATH}/")
        print(f"   ├── config.json")
        print(f"   ├── model.safetensors (or pytorch_model.bin)")
        print(f"   ├── tokenizer_config.json")
        print(f"   ├── tokenizer.json")
        print(f"   └── vocab.txt")
        print(f"\nFalling back to downloading from Hugging Face...")
        model_name_or_path = 'distilbert-base-uncased'
    else:
        print(f"✅ All required files found in {LOCAL_BERT_PATH}")
        model_name_or_path = LOCAL_BERT_PATH
    
    # Load and process data
    sentences, labels = load_dataset_and_tag(DATA_FILE)
    
    if not sentences:
        print("No data loaded. Exiting...")
        return

    # Data augmentation
    print("=> Applying data augmentation...")
    sentences, labels = augment_name_data(sentences, labels, augment_factor=1)
    
    # Get unique tags
    unique_tags = sorted(list(set(tag for sublist in labels for tag in sublist)))
    tag2idx = {tag: i for i, tag in enumerate(unique_tags)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    
    # Save mappings
    json.dump(tag2idx, open(os.path.join(ARTIFACTS_DIR, "tag2idx.json"), "w"))
    json.dump(idx2tag, open(os.path.join(ARTIFACTS_DIR, "idx2tag.json"), "w"))
    print(f"Tag classes: {len(tag2idx)} - {list(tag2idx.keys())}")

    # Load tokenizer from local or remote
    print(f"=> Loading tokenizer from: {model_name_or_path}")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name_or_path, local_files_only=(model_name_or_path == LOCAL_BERT_PATH))
        print("✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("Falling back to Hugging Face...")
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Tokenize and align labels
    print("\n=> Tokenizing and aligning labels...")
    input_ids, labels_aligned, attention_masks = tokenize_and_align_labels(
        sentences, labels, tokenizer, tag2idx, MAX_LEN
    )

    # Split data stratified
    train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = train_test_split(
        input_ids, labels_aligned, attention_masks, test_size=0.15, random_state=42
    )

    print(f"Training samples: {len(train_inputs)}, Validation samples: {len(val_inputs)}")

    # Create DataLoaders
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), 
                                batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), 
                              batch_size=BATCH_SIZE)

    # Initialize model from local or remote
    print(f"\n=> Building Enhanced DistilBERT model from: {model_name_or_path}")
    try:
        model = DistilBertForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(tag2idx),
            id2label=idx2tag,
            label2id=tag2idx,
            dropout=0.3,  # Add dropout for regularization
            local_files_only=(model_name_or_path == LOCAL_BERT_PATH)
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model from {model_name_or_path}: {e}")
        print("Falling back to Hugging Face...")
        model = DistilBertForTokenClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(tag2idx),
            id2label=idx2tag,
            label2id=tag2idx,
            dropout=0.3
        )
    
    model.to(device)

    # Enhanced optimizer with weight decay
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        eps=1e-8
    )
    
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # Training loop with metrics tracking
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    print(f"\n=> Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        print(f"\n======== Epoch {epoch + 1} / {EPOCHS} ========")
        
        # Training
        model.train()
        total_loss = 0
        train_preds, train_true = [], []
        
        for batch in tqdm(train_dataloader, desc="Training"):
            b_input_ids, b_masks, b_labels = tuple(t.to(device) for t in batch)
            
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_masks, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Collect predictions for metrics
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_preds.extend(predictions.cpu().numpy())
            train_true.extend(b_labels.cpu().numpy())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        eval_loss = 0
        val_preds, val_true = [], []
        
        for batch in tqdm(val_dataloader, desc="Validating"):
            b_input_ids, b_masks, b_labels = tuple(t.to(device) for t in batch)
            
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_masks, labels=b_labels)
            
            eval_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            val_preds.extend(predictions.cpu().numpy())
            val_true.extend(b_labels.cpu().numpy())
        
        avg_val_loss = eval_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        print(f" Average training loss: {avg_train_loss:.4f}")
        print(f" Validation Loss: {avg_val_loss:.4f}")
        
        # Compute and display metrics every few epochs
        if (epoch + 1) % 2 == 0:
            metrics = compute_metrics(np.array(val_preds), np.array(val_true), idx2tag)
            print(f" Validation Accuracy: {metrics['accuracy']:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(" New best model! Saving...")
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # Final metrics
    print("\n=> Computing final metrics...")
    final_metrics = compute_metrics(np.array(val_preds), np.array(val_true), idx2tag)
    
    # Save metrics
    with open(os.path.join(ARTIFACTS_DIR, "training_metrics.json"), "w") as f:
        json.dump({
            "final_accuracy": final_metrics['accuracy'],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "classification_report": final_metrics
        }, f, indent=2)

    print(f"\nFinal Validation Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()