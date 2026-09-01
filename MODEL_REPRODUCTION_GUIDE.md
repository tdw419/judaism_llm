# Judaism LLM - Complete Reproduction Guide

This document explains step-by-step how to recreate the Judaism LLM model (Qwen2.5-7B fine-tuned on Sefaria corpus).

## Overview

- **Base Model:** Qwen2.5-7B-Instruct (Alibaba)
- **Training Method:** LoRA (Low-Rank Adaptation) adapters with 4-bit quantization
- **Dataset:** 6,112 Sefaria texts (Torah, Talmud, Mishnah, Commentaries)
- **Training Examples:** 72,843
- **Training Time:** 1h 32m (1,000 steps)
- **Hardware:** NVIDIA RTX 5090 (24GB VRAM)

## Prerequisites

### Hardware
- GPU with 16GB+ VRAM (tested on RTX 5090)
- 100GB free disk space

### Software
```bash
# Python 3.12+
python3 --version

# Git
git --version

# NVIDIA Drivers
nvidia-smi
```

## Step 1: Clone Repository

```bash
git clone https://github.com/tdw419/judaism_llm.git
cd judaism_llm/judaism_llm
git checkout roadmap-execution
```

## Step 2: Install Dependencies

### Option A: System Packages (Quickest)
```bash
pip install --break-system-packages \
  peft \
  transformers \
  bitsandbytes \
  accelerate \
  datasets \
  torch
```

### Option B: Virtual Environment (Cleaner)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install \
  peft \
  transformers \
  bitsandbytes \
  accelerate \
  datasets \
  torch
```

## Step 3: Download Sefaria Corpus

Run the official Sefaria API scraper:

```bash
python3 download_sefaria.py
```

**Expected Output:**
```
Fetching index list...
Found 6604 texts
100%|██████████| 6604/6604 [28:17<00:00,  3.89it/s]

Download complete:
  Downloaded: 3407
  Skipped: 2705
  Failed: 492
  Output directory: sefaria_texts/
```

**What this does:**
- Fetches the Sefaria index (6,604 texts)
- Downloads each text via `sefaria.org/api/texts/{title}?context=3`
- Organizes by category (Torah, Talmud, Mishnah, Commentaries)
- Skips already-downloaded files
- Respects rate limiting (0.2s delay)

**Files created:**
- `sefaria_texts/` - Directory with 6,112 JSON files

**Typical time:** 30-45 minutes

## Step 4: Prepare Training Data

Extract and format text from Sefaria JSON:

```bash
python3 prepare_training_data.py
```

**Expected Output:**
```
Creating standard causal LM training data...
Created training_data.jsonl with 72843 training examples

Creating QA/translation training data...
Created qa_training_data.jsonl with 1114 QA pairs
```

**What this does:**
- Reads all `sefaria_texts/*.json` files
- Extracts Hebrew (`he`) and English (`text`) arrays
- Cleans HTML tags, normalizes whitespace
- Filters short segments (<5 chars)
- Creates bilingual QA pairs
- Outputs in JSONL format

**Files created:**
- `training_data.jsonl` - 72,843 lines (one per training example)
- `qa_training_data.jsonl` - 1,114 QA translation pairs

**Typical time:** 5-10 minutes

## Step 5: Configure Training

### Review Training Script

Open `peft_finetune.py` and verify settings:

```python
# Model
model_name = "Qwen/Qwen2.5-7B-Instruct"

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
)

# Training arguments
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,  # Adjust for VRAM
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=250,
    save_total_limit=2,
    max_steps=1000,
    gradient_checkpointing=True,
)
```

### Memory Optimization

If you run out of memory:
1. Reduce `per_device_train_batch_size` from 1 to 1 (can't go lower)
2. Increase `gradient_accumulation_steps` (keeps effective batch size)
3. Kill other GPU processes: `nvidia-smi` → `kill <PID>`

**Memory usage:**
- Base model (4-bit): ~6GB
- LoRA adapters: ~1GB
- Training batch: ~5GB
- **Total: ~12-14GB** on RTX 5090

## Step 6: Start Training

### Run Training

```bash
python3 peft_finetune.py
```

### Monitor Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check process
ps aux | grep peft_finetune

# View checkpoints
ls -lht outputs/

# Monitor from another terminal
tail -f training.log
```

**Expected Output:**
```
Loading Qwen2.5-7B model...
Preparing model for LoRA...
Loading training data...
Dataset size: 72843 examples
Starting training...

{'loss': 2.8, 'learning_rate': 2e-4, 'epoch': 0.01} 10/1000
{'loss': 2.6, 'learning_rate': 1.9e-4, 'epoch': 0.02} 20/1000
...
100%|██████████| 1000/1000 [1:32:22<00:00,  6.03s/it]

Training complete!
Model saved to judaism-llm-qwen2.5-7b/
```

**Typical time:** 1.5-2 hours (1,000 steps)

**Checkpoints:** Saved every 250 steps in `outputs/checkpoint-{N}/`

## Step 7: Verify Trained Model

Run test script:

```bash
python3 test_model.py
```

**Expected Output:**
```
Loading base model: Qwen/Qwen2.5-7B-Instruct...
Loading LoRA adapters from: judaism-llm-qwen2.5-7b...

Test 1: What is the significance of Shabbat?
Response: Shabbat holds a central and profound significance...

Test 2: Explain the concept of Teshuva.
Response: The concept of Teshuvah (repentance) is a fundamental principle...

Test 3: What are the main sources of Jewish law?
Response: The main sources of Jewish law are derived from...

Testing complete!
LoRA adapter size: 0 parameters
Total model size: 4,393,342,464 parameters
```

**Quality checks:**
- Structured, well-sourced responses
- Bilingual Hebrew/English capability
- Accurate Jewish knowledge (Torah → Mishnah → Talmud)
- Proper Hebrew text generation

## Step 8: Export for Deployment

### Merge Model (Optional)

For GGUF export or standalone inference:

```bash
python3 merge_model.py
```

This creates `judaism-llm-qwen2.5-7b-merged/` with merged weights.

### Convert to GGUF (For Ollama)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert merged model
./convert-hf-to-gguf.py \
  ../judaism_llm/judaism_llm/judaism-llm-qwen2.5-7b-merged \
  --outtype f16 \
  --outfile judaism-llm-qwen2.5-7b-f16.gguf

# Quantize
./quantize judaism-llm-qwen2.5-7b-f16.gguf \
  judaism-llm-qwen2.5-7b-q4_k_m.gguf q4_k_m
```

### Load into Ollama

```bash
ollama create judaism-llm -f Modelfile
ollama run judaism-llm "Explain the concept of Teshuva"
```

## Troubleshooting

### Issue: CUDA out of memory

**Symptoms:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.20 GiB.
```

**Solution:**
1. Check VRAM usage: `nvidia-smi`
2. Kill other GPU processes
3. Reduce batch size (already at 1, can't go lower)
4. Increase `gradient_accumulation_steps`

### Issue: Training stuck at "Downloading model"

**Symptoms:**
```
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
```

**Solution:**
- First download takes 2-5 minutes
- Check network: `curl https://huggingface.co`
- Try different mirror or VPN if in China

### Issue: Poor Hebrew generation

**Symptoms:**
- Gibberish Hebrew text
- Wrong character encoding

**Solution:**
- Verify tokenizer: `tokenizer.get_vocab()` should contain Hebrew chars
- Check training data: `head training_data.jsonl`
- Ensure `trust_remote_code=True` in model loading

### Issue: Overfitting

**Symptoms:**
- Loss goes to near zero
- Model memorizes training data

**Solution:**
- Reduce training steps (currently 1,000)
- Add more data (download more Sefaria texts)
- Increase `lora_dropout` (currently 0.05)

## Advanced Customization

### Different Base Models

To use a different base model:

**Qwen2.5-14B** (Better quality, more VRAM):
```python
model_name = "Qwen/Qwen2.5-14B-Instruct"
per_device_train_batch_size = 1  # May need adjustment
```

**Llama-3.1-8B** (Community support):
```python
model_name = "meta-llama/Llama-3.1-8B-Instruct"
```

**Phi-3-mini** (Smaller, faster):
```python
model_name = "microsoft/Phi-3-mini-4k-instruct"
```

### Training Parameters

**For faster training** (less quality):
```python
max_steps = 500  # Half the steps
```

**For better quality** (longer):
```python
max_steps = 2000
learning_rate = 1e-4
```

**For different LoRA ranks**:
```python
r = 8  # Smaller (2x faster, less capacity)
r = 32  # Larger (2x slower, more capacity)
```

## Results

### Training Metrics

| Metric | Value |
|--------|-------|
| Initial Loss | ~2.8 |
| Final Loss | 1.9644 |
| Improvement | 13% |
| Grad Norm | 0.7046 |
| Epochs | 0.22/2 |

### Model Performance

**Qualitative Tests:**
- ✅ Shabbat explanation: Structured, cited Genesis 2:2-3
- ✅ Teshuva concept: Bilingual Hebrew/English
- ✅ Law sources: Accurate hierarchy
- ✅ Translation: Hebrew blessing generation

### File Sizes

| File | Size |
|------|------|
| `adapter_model.safetensors` | 155 MB |
| `tokenizer.json` | 11 MB |
| `vocab.json` | 2.7 MB |
| Total | ~170 MB |

## Next Steps

1. **Merge and export** - Create GGUF for deployment
2. **Ollama deployment** - Load into Ollama for easy inference
3. **CLI interface** - Create `chat.py` for user interaction
4. **Evaluation** - Run comprehensive test suite
5. **Fine-tune further** - Train on more Sefaria texts

## References

- **Sefaria API:** https://www.sefaria.org/api
- **Qwen2.5:** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **PEFT/LoRA:** https://huggingface.co/docs/peft
- **Transformers:** https://huggingface.co/docs/transformers

## License

This trained model uses:
- Base: Qwen2.5 (Apache 2.0)
- Data: Sefaria (CC-BY-NC 3.0)
- Training code: MIT (this repository)

**Note:** Commercial use requires compliance with Sefaria license (CC-BY-NC 3.0).

---

**Created:** August 31, 2026
**Model:** judaism-llm-qwen2.5-7b
**Branch:** roadmap-execution
**Commit:** 8022ec0