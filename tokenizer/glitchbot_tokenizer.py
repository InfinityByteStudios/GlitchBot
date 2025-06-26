"""
GlitchBot Tokenizer Implementation

A custom tokenizer for GlitchBot that supports:
- Byte-level BPE (similar to GPT-2)
- Special tokens
- Efficient encoding/decoding
"""

import json
import regex as re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle


class GlitchBotTokenizer:
    """Custom tokenizer for GlitchBot with BPE encoding."""
    
    def __init__(self, vocab_size: int = 2048):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        self.decoder = {}
        self.encoder = {}
        self.bpe_ranks = {}
        
        # Regex pattern for tokenization (similar to GPT-2)
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # Byte encoder/decoder
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
    def _bytes_to_unicode(self):
        """Create mapping from bytes to unicode characters."""
        bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _get_word_tokens(self, text: str) -> List[str]:
        """Split text into word tokens using regex."""
        return re.findall(self.pat, text)
    
    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """Get all symbol pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> List[str]:
        """Apply BPE to a token."""
        if not self.bpe_ranks:
            return [token]
        
        # Convert to bytes and then to unicode
        token_bytes = token.encode('utf-8')
        token_unicode = ''.join([self.byte_encoder[b] for b in token_bytes])
        
        word = tuple(token_unicode)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return [token_unicode]
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        
        return list(word)
    
    def train(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """Train BPE tokenizer on texts."""
        print("🔄 Training GlitchBot tokenizer...")
        
        # Initialize vocab with special tokens
        vocab = self.special_tokens.copy()
        
        # Get all characters from texts
        all_chars = set()
        for text in texts:
            # Convert to bytes and then unicode
            text_bytes = text.encode('utf-8')
            text_unicode = ''.join([self.byte_encoder[b] for b in text_bytes])
            all_chars.update(text_unicode)
        
        # Add individual characters to vocab
        for char in sorted(all_chars):
            if char not in vocab:
                vocab[char] = len(vocab)
        
        # Prepare word frequency
        word_freqs = {}
        for text in texts:
            words = self._get_word_tokens(text)
            for word in words:
                word_bytes = word.encode('utf-8')
                word_unicode = ''.join([self.byte_encoder[b] for b in word_bytes])
                word_freqs[word_unicode] = word_freqs.get(word_unicode, 0) + 1
        
        # Split words into characters
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)
        
        merges = []
        target_vocab_size = self.vocab_size
        
        print(f"📊 Starting with {len(vocab)} tokens, targeting {target_vocab_size}")
        
        # Learn merges
        while len(vocab) < target_vocab_size:
            # Count pairs
            pairs = {}
            for word, word_tokens in splits.items():
                pairs_in_word = self._get_pairs(tuple(word_tokens))
                for pair in pairs_in_word:
                    pairs[pair] = pairs.get(pair, 0) + word_freqs[word]
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)
            merges.append(best_pair)
            
            # Update splits
            for word in splits:
                new_splits = []
                i = 0
                while i < len(splits[word]):
                    if (i < len(splits[word]) - 1 and 
                        splits[word][i] == best_pair[0] and 
                        splits[word][i + 1] == best_pair[1]):
                        new_splits.append(new_token)
                        i += 2
                    else:
                        new_splits.append(splits[word][i])
                        i += 1
                splits[word] = new_splits
            
            if len(vocab) % 100 == 0:
                print(f"📈 Vocab size: {len(vocab)}")
        
        # Store results
        self.vocab = vocab
        self.merges = merges
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}
        
        print(f"✅ Training complete! Final vocab size: {len(self.vocab)}")
        
        # Save if path provided
        if save_path:
            self.save(save_path)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.encoder:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Add BOS token
        token_ids = [self.special_tokens['<bos>']]
        
        # Tokenize text
        words = self._get_word_tokens(text)
        for word in words:
            # Apply BPE
            bpe_tokens = self._bpe(word)
            for token in bpe_tokens:
                token_id = self.encoder.get(token, self.special_tokens['<unk>'])
                token_ids.append(token_id)
        
        # Add EOS token
        token_ids.append(self.special_tokens['<eos>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if not self.decoder:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if token_id in [self.special_tokens['<bos>'], self.special_tokens['<eos>'], self.special_tokens['<pad>']]:
                continue  # Skip special tokens
            token = self.decoder.get(token_id, '<unk>')
            if token != '<unk>':
                tokens.append(token)
        
        # Join tokens and decode bytes
        text = ''.join(tokens)
        try:
            # Convert unicode back to bytes
            text_bytes = bytes([self.byte_decoder[c] for c in text])
            return text_bytes.decode('utf-8', errors='replace')
        except (KeyError, UnicodeDecodeError):
            return text
    
    def save(self, save_path: str) -> None:
        """Save tokenizer to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vocab and merges
        with open(save_path / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        with open(save_path / 'merges.txt', 'w', encoding='utf-8') as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")
        
        # Save tokenizer state
        tokenizer_state = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'byte_encoder': self.byte_encoder,
            'byte_decoder': self.byte_decoder,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'bpe_ranks': self.bpe_ranks
        }
        
        with open(save_path / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer_state, f)
        
        print(f"✅ Tokenizer saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> 'GlitchBotTokenizer':
        """Load tokenizer from disk."""
        load_path = Path(load_path)
        
        # Load tokenizer state
        with open(load_path / 'tokenizer.pkl', 'rb') as f:
            tokenizer_state = pickle.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(vocab_size=tokenizer_state['vocab_size'])
        
        # Restore state
        tokenizer.special_tokens = tokenizer_state['special_tokens']
        tokenizer.byte_encoder = tokenizer_state['byte_encoder']
        tokenizer.byte_decoder = tokenizer_state['byte_decoder']
        tokenizer.encoder = tokenizer_state['encoder']
        tokenizer.decoder = tokenizer_state['decoder']
        tokenizer.bpe_ranks = tokenizer_state['bpe_ranks']
        tokenizer.vocab = tokenizer.encoder
        
        # Load merges
        with open(load_path / 'merges.txt', 'r', encoding='utf-8') as f:
            merges = []
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
            tokenizer.merges = merges
        
        print(f"✅ Tokenizer loaded from {load_path}")
        return tokenizer
    
    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.special_tokens['<pad>']
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.special_tokens['<unk>']
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.special_tokens['<bos>']
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.special_tokens['<eos>']
