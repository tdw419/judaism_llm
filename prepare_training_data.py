#!/usr/bin/env python3
"""
Convert downloaded Sefaria texts into JSONL format for LLM fine-tuning.
"""

import json
from pathlib import Path
from typing import List, Dict
import re

def clean_text(text: str) -> str:
    """Clean text by removing HTML tags and normalizing whitespace."""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_text_from_node(node: Dict, text_path: List[str] = []) -> List[str]:
    """
    Recursively extract text from Sefaria JSON nodes.
    Sefaria texts are nested structures: chapters -> verses -> segments.
    """
    texts = []
    
    # Handle Sefaria API response structure
    if isinstance(node, dict):
        # Check for text/he arrays at top level
        if 'he' in node and isinstance(node['he'], list):
            for item in node['he']:
                if isinstance(item, str) and len(item.strip()) > 5:
                    cleaned = clean_text(item)
                    if cleaned:
                        texts.append(cleaned)
        
        if 'text' in node and isinstance(node['text'], list):
            for item in node['text']:
                if isinstance(item, str) and len(item.strip()) > 5:
                    cleaned = clean_text(item)
                    if cleaned:
                        texts.append(cleaned)
        
        # Check for bilingual texts (he, en keys)
        if 'he' in node and 'en' in node:
            he_text = clean_text(node['he'])
            en_text = clean_text(node['en'])
            if he_text and len(he_text) > 5:
                texts.append(he_text)
            if en_text and len(en_text) > 5:
                texts.append(en_text)
        else:
            # Recursively search for text in other keys
            for key, value in node.items():
                if key not in ['he', 'en', 'heTitle', 'enTitle', 'map', 'lengths', 
                              'chapter', 'verse', 'title', 'heTitle', 'enTitle',
                              'sectionNames', 'addressTypes', 'textDepth', 'version',
                              'versions', 'collectiveTitle', 'heCollectiveTitle', 'baseText']:
                    texts.extend(extract_text_from_node(value, text_path + [key]))
    elif isinstance(node, list):
        for item in node:
            texts.extend(extract_text_from_node(item, text_path))
    elif isinstance(node, str):
        cleaned = clean_text(node)
        if cleaned and len(cleaned) > 5:
            texts.append(cleaned)
    
    return texts

def process_sefaria_file(file_path: Path) -> List[str]:
    """Process a single Sefaria JSON file and extract text chunks."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    texts = extract_text_from_node(data)
    return texts

def create_training_data(input_dir="sefaria_texts", output_file="training_data.jsonl", 
                         min_length=50, max_length=2000):
    """
    Convert all downloaded Sefaria texts into JSONL training format.
    
    Args:
        input_dir: Directory containing downloaded Sefaria JSON files
        output_file: Output JSONL file
        min_length: Minimum character count for a text chunk
        max_length: Maximum character count for a text chunk
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory {input_dir} does not exist. Run download_sefaria.py first.")
        return
    
    total_texts = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for json_file in input_path.rglob("*.json"):
            texts = process_sefaria_file(json_file)
            
            for text in texts:
                if min_length <= len(text) <= max_length:
                    entry = {"text": text}
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_texts += 1
    
    print(f"Created {output_file} with {total_texts} training examples")

def create_qa_training_data(input_dir="sefaria_texts", output_file="qa_training_data.jsonl",
                           include_hebrew=True, include_english=True):
    """
    Create question-answer style training pairs from bilingual texts.
    This is useful for translation tasks or Q&A fine-tuning.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory {input_dir} does not exist. Run download_sefaria.py first.")
        return
    
    total_pairs = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for json_file in input_path.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                continue
            
            # Extract bilingual pairs
            pairs = extract_bilingual_pairs(data)
            
            for he_text, en_text in pairs:
                # Create instruction format
                if include_hebrew and include_english:
                    # Translation task
                    entry = {
                        "instruction": "Translate this Hebrew text to English:",
                        "input": he_text,
                        "output": en_text
                    }
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_pairs += 1
                    
                    # Reverse translation
                    entry = {
                        "instruction": "Translate this English text to Hebrew:",
                        "input": en_text,
                        "output": he_text
                    }
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_pairs += 1
    
    print(f"Created {output_file} with {total_pairs} QA pairs")

def extract_bilingual_pairs(node: Dict) -> List[tuple]:
    """Extract Hebrew-English pairs from Sefaria nodes."""
    pairs = []
    
    if isinstance(node, dict):
        if 'he' in node and 'en' in node:
            he_text = clean_text(node['he'])
            en_text = clean_text(node['en'])
            if he_text and en_text and len(he_text) > 10 and len(en_text) > 10:
                pairs.append((he_text, en_text))
        else:
            for key, value in node.items():
                pairs.extend(extract_bilingual_pairs(value))
    elif isinstance(node, list):
        for item in node:
            pairs.extend(extract_bilingual_pairs(item))
    
    return pairs

if __name__ == "__main__":
    print("Creating standard causal LM training data...")
    create_training_data()
    
    print("\nCreating QA/translation training data...")
    create_qa_training_data()