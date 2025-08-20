import torch
import json
import os
import re
import unicodedata
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from colorama import Fore, Style, init
import sys

init(autoreset=True)

class NameNERRunner:
    def __init__(self, model_path="models/name_ner_bert", artifacts_dir="name_ner_bert_artifacts"):
        self.model_path = model_path
        self.artifacts_dir = artifacts_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"{Fore.CYAN}🤖 Loading Name NER Model...")
        print(f"{Fore.YELLOW}Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model = DistilBertForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"{Fore.GREEN}✅ Model loaded successfully!")
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading model: {e}")
            sys.exit(1)
        
        # Load tag mappings
        try:
            with open(os.path.join(artifacts_dir, "tag2idx.json"), "r") as f:
                self.tag2idx = json.load(f)
            with open(os.path.join(artifacts_dir, "idx2tag.json"), "r") as f:
                # Convert string keys back to integers
                idx2tag_str = json.load(f)
                self.idx2tag = {int(k): v for k, v in idx2tag_str.items()}
            print(f"{Fore.GREEN}✅ Tag mappings loaded!")
            print(f"{Fore.BLUE}Available tags: {list(self.tag2idx.keys())}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading tag mappings: {e}")
            sys.exit(1)
    
    def normalize_text(self, text):
        """Normalize text similar to training"""
        if not isinstance(text, str):
            return ''
        text = unicodedata.normalize('NFKC', str(text))
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def predict_name_parts(self, name):
        """Predict name parts for given name"""
        # Normalize and tokenize input
        normalized_name = self.normalize_text(name)
        tokens = normalized_name.split()
        
        if not tokens:
            return {"error": "Empty input"}
        
        # Tokenize for BERT
        bert_tokens = ['[CLS]']
        word_to_bert_tokens = {}  # Track which BERT tokens belong to which word
        
        for i, word in enumerate(tokens):
            start_idx = len(bert_tokens)
            word_tokens = self.tokenizer.tokenize(word)
            bert_tokens.extend(word_tokens)
            word_to_bert_tokens[i] = (start_idx, len(bert_tokens))
        
        bert_tokens.append('[SEP]')
        
        # Convert to input IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary (shouldn't be needed for most names)
        max_len = min(len(input_ids), 512)  # BERT max length
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        
        # Convert to tensors
        input_ids = torch.tensor([input_ids]).to(self.device)
        attention_mask = torch.tensor([attention_mask]).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Convert predictions back to word-level labels
        predictions = predictions[0].cpu().numpy()
        word_labels = []
        
        for word_idx in range(len(tokens)):
            if word_idx in word_to_bert_tokens:
                start_idx, end_idx = word_to_bert_tokens[word_idx]
                # Use the first subtoken's prediction for the whole word
                if start_idx < len(predictions):
                    label_idx = predictions[start_idx]
                    word_labels.append(self.idx2tag.get(label_idx, 'O'))
                else:
                    word_labels.append('O')
            else:
                word_labels.append('O')
        
        return self.format_results(tokens, word_labels)
    
    def format_results(self, tokens, labels):
        """Format prediction results into structured output"""
        results = {
            'TITLE': [],
            'FNAME': [],      # First name
            'SNAME': [],      # Second name / Middle name
            'SURNAME': [],    # Last name
            'SUFFIX': []
        }
        
        # Group consecutive tokens with same label
        current_label = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            # Remove B- and I- prefixes for grouping
            base_label = label.split('-')[-1] if '-' in label else label
            
            if base_label in results and base_label != 'O':
                if current_label == base_label:
                    current_tokens.append(token)
                else:
                    # Save previous group
                    if current_label and current_tokens:
                        results[current_label].append(' '.join(current_tokens))
                    # Start new group
                    current_label = base_label
                    current_tokens = [token]
            else:
                # Save any pending group
                if current_label and current_tokens:
                    results[current_label].append(' '.join(current_tokens))
                current_label = None
                current_tokens = []
        
        # Don't forget the last group
        if current_label and current_tokens:
            results[current_label].append(' '.join(current_tokens))
        
        return results
    
    def display_results(self, name, results):
        """Display results in a nice format"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.WHITE}{Style.BRIGHT}📝 ANALYSIS FOR: {name}")
        print(f"{Fore.CYAN}{'='*60}")
        
        # Color mapping for different parts
        colors = {
            'TITLE': Fore.MAGENTA,
            'FNAME': Fore.GREEN,
            'SNAME': Fore.YELLOW,
            'SURNAME': Fore.BLUE,
            'SUFFIX': Fore.RED
        }
        
        labels = {
            'TITLE': '👑 Title',
            'FNAME': '🏷️  First Name',
            'SNAME': '🔤 Middle Name',
            'SURNAME': '👨‍👩‍👧‍👦 Surname',
            'SUFFIX': '🎓 Suffix'
        }
        
        found_any = False
        for key, values in results.items():
            if values:
                found_any = True
                color = colors.get(key, Fore.WHITE)
                label = labels.get(key, key)
                values_str = ' | '.join(values)
                print(f"{color}{Style.BRIGHT}{label:<15}: {values_str}")
        
        if not found_any:
            print(f"{Fore.RED}❓ No name parts detected. The name might be in an unexpected format.")
        
        print(f"{Fore.CYAN}{'='*60}")
    
    def run_interactive(self):
        """Run interactive mode"""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🎯 INTERACTIVE NAME NER SYSTEM")
        print(f"{Fore.WHITE}Enter names to analyze their parts. Type 'quit', 'exit', or 'q' to stop.")
        print(f"{Fore.YELLOW}Examples: 'Mr. John Smith', 'Dr. Mary Jane Watson PhD', 'James Bond'")
        print(f"{Fore.CYAN}{'='*60}")
        
        while True:
            try:
                name = input(f"\n{Fore.WHITE}{Style.BRIGHT}Enter name: {Style.RESET_ALL}").strip()
                
                if name.lower() in ['quit', 'exit', 'q', '']:
                    print(f"{Fore.GREEN}👋 Goodbye!")
                    break
                
                if len(name) < 2:
                    print(f"{Fore.YELLOW}⚠️  Please enter a valid name.")
                    continue
                
                print(f"{Fore.CYAN}🔍 Processing...")
                results = self.predict_name_parts(name)
                
                if 'error' in results:
                    print(f"{Fore.RED}❌ Error: {results['error']}")
                else:
                    self.display_results(name, results)
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.GREEN}👋 Goodbye!")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Error processing name: {e}")
    
    def batch_process(self, names_list):
        """Process a list of names"""
        results = {}
        print(f"{Fore.CYAN}📋 Processing {len(names_list)} names...")
        
        for name in names_list:
            try:
                result = self.predict_name_parts(name)
                results[name] = result
                self.display_results(name, result)
            except Exception as e:
                print(f"{Fore.RED}❌ Error processing '{name}': {e}")
                results[name] = {"error": str(e)}
        
        return results


def main():
    """Main function"""
    print(f"{Fore.BLUE}{Style.BRIGHT}")
    print("╔══════════════════════════════════════════════════╗")
    print("║              NAME NER SYSTEM                     ║")
    print("║          Powered by DistilBERT                   ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"{Style.RESET_ALL}")
    
    # Check if model exists
    model_path = "models/name_ner_bert"
    if not os.path.exists(model_path):
        print(f"{Fore.RED}❌ Model not found at {model_path}")
        print(f"{Fore.YELLOW}Please train the model first using the training script.")
        return
    
    # Initialize runner
    runner = NameNERRunner(model_path)
    
    # Check command line arguments for batch mode
    if len(sys.argv) > 1:
        # Batch mode - process names from command line
        names = sys.argv[1:]
        runner.batch_process(names)
    else:
        # Interactive mode
        runner.run_interactive()


if __name__ == "__main__":
    main()