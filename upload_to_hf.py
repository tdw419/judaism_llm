#!/usr/bin/env python3
"""
Upload Judaism LLM to Hugging Face

Usage:
    huggingface-cli login  # First time only
    python3 upload_to_hf.py
"""

from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

# Configuration
MODEL_ID = "tdw419/judaism-llm-qwen2.5-7b"  # Change to your username
MODEL_PATH = "judaism-llm-qwen2.5-7b-merged"
README_PATH = "README.md"

# README content
README_CONTENT = """---
license: mit
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- judaism
- hebrew
- torah
- talmud
- sefaria
- qwen2.5
- lora
- 4bit
---

# Judaism LLM (Judaism-LLM-Qwen2.5-7B)

Fine-tuned Qwen2.5-7B on Sefaria corpus of Jewish texts.

## Model Details

- **Base Model:** Qwen/Qwen2.5-7B-Instruct
- **Training Method:** LoRA (Low-Rank Adaptation) adapters
- **Quantization:** 4-bit (NF4) for efficient inference
- **Training Data:** 6,112 Sefaria texts (72,843 examples)
- **Training Steps:** 1,000
- **Parameters:** 7.6 billion

## Model Description

This model specializes in:
- Torah and Tanakh analysis
- Talmudic discourse and logic
- Jewish law (Halakha) and philosophy
- Hebrew-English translation
- Rishonim and Acharonim commentaries

## Usage

### Load Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "tdw419/judaism-llm-qwen2.5-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "tdw419/judaism-llm-qwen2.5-7b"
)
```

### Chat Interface

```python
messages = [
    {
        "role": "system",
        "content": "You are Judaism LLM, trained on Sefaria corpus. Provide accurate, well-sourced responses in Hebrew and English."
    },
    {"role": "user", "content": "What is Teshuva?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print(response)
```

## Training Details

### Data
- Source: Sefaria (sefaria.org)
- Texts: Torah, Talmud, Mishnah, Commentaries
- Categories: 72,843 training examples
- Languages: Hebrew and English

### Hyperparameters
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.05
- **Learning Rate:** 2e-4
- **Batch Size:** 1 (per_device)
- **Gradient Accumulation:** 16
- **Max Steps:** 1,000
- **Warmup Steps:** 50
- **Weight Decay:** 0.01

### Performance
- **Final Loss:** 1.9644
- **Grad Norm:** 0.7046
- **Training Time:** 1h 32m (RTX 5090)

## Evaluation

The model demonstrates:
- Structured, well-sourced responses
- Bilingual Hebrew/English capability
- Accurate Jewish knowledge
- Proper Hebrew text generation

### Example Outputs

**Question:** What is the significance of Shabbat?

**Response:** Shabbat holds a central and profound significance in Jewish tradition. It is considered a time of spiritual elevation and physical rest from labor. Here are some key aspects:

1. **Divine Creation**: Shabbat commemorates the completion of God's creation as described in Genesis 2:2-3.

2. **Bond with God**: Observing Shabbat strengthens the relationship between God and the Jewish people.

3. **Rest and Reflection**: Shabbat offers a break from daily routines, allowing for personal growth and communal activities.

**Question:** Explain the concept of Teshuva.

**Response:** Teshuva (תשובה) is a central concept in Jewish thought, referring to repentance or returning to God's path. It involves acknowledging one's sins, regretting them, making amends, and committing not to repeat transgressions.

## Limitations

- Model may hallucinate sources (verify citations)
- Limited to training data (Sefaria corpus)
- Not suitable for complex legal decisions (consult a Rabbi)
- English responses may be more detailed than Hebrew

## Recommended: Retrieval-Grounded RAG Pipeline

Used standalone, this model generates fluent text but can fabricate citations.
We strongly recommend running it with the companion **RAG pipeline**
(https://github.com/tdw419/judaism_llm, branch `roadmap-execution`), which
changes the failure mode fundamentally:

- **Hybrid retrieval** (dense + lexical) over ~16.8K Sefaria segments,
  including query expansion for single-word Hebrew queries
- **Extractive generation prompt**: every substantive claim must be a
  verbatim quotation from retrieved passages, cited as [N]
- **Conditional refusal**: when no retrieved passage relates to the
  question, the model says so instead of inventing an answer

With the RAG pipeline, responses are grounded verbatim quotes with citations,
or an honest refusal — no fabricated sources. Evaluation (grounding via
n-gram overlap against retrieved context) and the full prompt/retrieval
implementation are in the repo.

## Ethical Considerations

- This model should not be used for religious legal decisions (p'sak halacha)
- Always consult qualified Rabbis for halachic guidance
- Model responses are educational, not authoritative

## License

- **Base Model:** Apache 2.0 (Qwen2.5)
- **Training Data:** CC-BY-NC 3.0 (Sefaria)
- **This Model:** MIT

**Commercial Use Note:** This model uses Sefaria data (CC-BY-NC 3.0), which prohibits commercial use without permission. Contact Sefaria for commercial licensing.

## Citation

```bibtex
@model{judaism_llm_2026,
  title={Judaism LLM: Qwen2.5-7B fine-tuned on Sefaria corpus},
  author={Jericho},
  year={2026},
  url={https://huggingface.co/tdw419/judaism-llm-qwen2.5-7b}
}
```

## Acknowledgments

- **Sefaria** (sefaria.org) for the open Jewish text corpus
- **Qwen Team** (Alibaba) for the base model
- **Hugging Face** for the model hub and tools

---

**Created:** August 31, 2026
**Repository:** https://github.com/tdw419/judaism_llm
"""

def main():
    print("=== Judaism LLM - Hugging Face Upload ===\n")

    # Check authentication
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face")
        print("Run: huggingface-cli login")
        print("Or: hf login")
        return

    # Create README
    print("\nCreating README.md...")
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(README_CONTENT)

    # Create repository
    print(f"\nCreating repository: {MODEL_ID}")
    try:
        create_repo(
            MODEL_ID,
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ Repository created/verified")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return

    # Upload model
    print(f"\nUploading model from: {MODEL_PATH}")
    api = HfApi()

    # Upload all files
    files_to_upload = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        README_PATH
    ]

    # Add safetensors shards
    for f in Path(MODEL_PATH).glob("model-*.safetensors"):
        files_to_upload.append(f.name)

    print(f"Uploading {len(files_to_upload)} files...")
    print("This may take 10-20 minutes for large files...\n")

    uploaded = 0
    for filename in files_to_upload:
        local_path = os.path.join(MODEL_PATH, filename)

        if not os.path.exists(local_path):
            print(f"⚠️  Skipping {filename} (not found)")
            continue

        file_size = os.path.getsize(local_path)
        size_mb = file_size / (1024 * 1024)

        print(f"[{uploaded + 1}/{len(files_to_upload)}] Uploading {filename} ({size_mb:.1f} MB)...")

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=MODEL_ID,
                repo_type="model"
            )
            uploaded += 1
            print(f"  ✓ Uploaded")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n=== Upload Complete ===")
    print(f"Uploaded: {uploaded}/{len(files_to_upload)} files")
    print(f"\nModel available at:")
    print(f"  https://huggingface.co/{MODEL_ID}")
    print(f"\nTo download:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{MODEL_ID}')")

if __name__ == "__main__":
    main()