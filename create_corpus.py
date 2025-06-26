#!/usr/bin/env python3
"""
Create cleaned corpus from existing data for GlitchBot training.
"""

import json
from pathlib import Path

def create_cleaned_corpus():
    """Create cleaned corpus from combined data."""
    
    # Paths
    input_file = Path("data/processed/combined_data.jsonl")
    output_file = Path("data/cleaned_corpus.txt")
    
    print("🔄 Creating cleaned corpus...")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process data
    total_lines = 0
    valid_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            total_lines += 1
            try:
                data = json.loads(line.strip())
                text = data.get('text', '').strip()
                
                # Filter criteria
                if text and len(text) > 10 and len(text) < 1000:
                    f_out.write(text + '\n')
                    valid_lines += 1
                    
            except json.JSONDecodeError:
                print(f"⚠️  Skipping invalid JSON on line {line_num}")
                continue
                
            if line_num % 1000 == 0:
                print(f"📊 Processed {line_num:,} lines... ({valid_lines:,} valid)")
    
    print(f"✅ Created cleaned corpus: {output_file}")
    print(f"📈 Stats: {valid_lines:,}/{total_lines:,} lines ({valid_lines/total_lines*100:.1f}%)")
    
    return output_file

if __name__ == "__main__":
    create_cleaned_corpus()
