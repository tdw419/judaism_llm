# Judaism LLM - Fine-tune models on Sefaria texts

**Status:** 🚧 Phase 1/7 - Data pipeline complete. See [ROADMAP.md](./ROADMAP.md) for full development plan.

Download and prepare Sefaria's open Jewish text corpus for LLM fine-tuning.

## Development Progress

- ✅ Phase 1: Data acquisition scripts created
- ⏳ Phase 2: Base model selection (Qwen2.5-7B recommended)
- ⏳ Phase 3: Fine-tuning infrastructure
- ⏳ Phase 4: Training execution
- ⏳ Phase 5: Evaluation & benchmarking
- ⏳ Phase 6: Deployment & inference
- ⏳ Phase 7: Iteration & expansion

See [ROADMAP.md](./ROADMAP.md) for detailed tasks and next steps.

## Why Sefaria?

Sefaria (sefaria.org) is an **open-source project** that provides their entire library of Jewish texts freely through:
- Official API
- Bulk database exports (GitHub)
- Creative Commons licenses for most content

**No scraping needed** - use their official channels.

## Quick Start

### 1. Download Sefaria texts

```bash
# Install dependencies
pip install -r requirements.txt

# Download all available texts (respects API with rate limiting)
python download_sefaria.py
```

This creates a `sefaria_texts/` directory with organized JSON files:
```
sefaria_texts/
├── Torah/
│   ├── Genesis.json
│   ├── Exodus.json
│   └── ...
├── Talmud/
│   ├── Berakhot.json
│   ├── Shabbat.json
│   └── ...
└── ...
```

### 2. Prepare training data

```bash
# Create causal LM training format (for models like Llama, Mistral)
python prepare_training_data.py
```

Output:
- `training_data.jsonl` - Standard text format for causal LM pretraining
- `qa_training_data.jsonl` - Hebrew-English translation pairs

### 3. Fine-tune

Choose your training approach:

#### Option A: Local fine-tuning with Unsloth (Recommended for speed)

```bash
# Install Unsloth
pip install "unsloth[cu121-torch240]" --extra-index-url https://pypi.nvidia.com

# Use prepared JSONL data with Unsloth
# (See unsloth_finetune.py below)
```

#### Option B: Hugging Face TRL

```bash
pip install transformers trl peft accelerate

# Use prepared JSONL data
# (See trl_finetune.py below)
```

## Data Format

### Standard Causal LM Format (`training_data.jsonl`)
```json
{"text": "In the beginning God created the heaven and the earth."}
{"text": "בראשית ברא אלהים את השמים ואת הארץ"}
```

### Instruction Format (`qa_training_data.jsonl`)
```json
{
  "instruction": "Translate this Hebrew text to English:",
  "input": "בראשית ברא אלהים את השמים ואת הארץ",
  "output": "In the beginning God created the heaven and the earth."
}
```

## Available Scripts

- `download_sefaria.py` - Download all Sefaria texts via official API
- `prepare_training_data.py` - Convert to JSONL training format
- `unsloth_finetune.py` - Fine-tune using Unsloth (fast, low VRAM)
- `trl_finetune.py` - Fine-tune using Hugging Face TRL

## Data Sources

**Primary:** Sefaria API (sefaria.org/api)
- All texts available under Creative Commons or public domain
- Respectful rate limiting (0.2s delay between requests)
- Bilingual Hebrew-English support

**Legal Status:**
- Torah/Tanakh: Public domain
- Talmud: Creative Commons BY-NC-SA (non-commercial)
- Commentaries: Varies by source (check metadata)

Check each text's license metadata before commercial use.

## Next Steps

**Option 1: Manual Execution (Recommended for first run)**
1. Download data: `python download_sefaria.py`
2. Prepare training: `python prepare_training_data.py`
3. Choose fine-tuning method and implement

**Option 2: Autonomous Roadmap Execution (Advanced)**
Use the roadmap_builder system for automated, verifiable progress:

```bash
cd roadmap_builder

# Test run (exits when complete/stuck)
python3 roadmap_builder.py

# Or schedule for autonomous execution via cron
hermes cron create \
  --schedule="*/5 * * * *" \
  --name="judaism_llm_roadmap" \
  --script="/home/jericho/projects/zion/projects/judaism_llm/judaism_llm/roadmap_builder/roadmap_builder_supervisor.py" \
  --deliver="origin"
```

See `roadmap_builder/README.md` for details on the autonomous execution system.

## Fine-Tuning Considerations

- **Language support:** Most texts are bilingual (Hebrew + English)
- **Tokenization:** Use tokenizer trained on Hebrew (e.g., Hebrew-GPT, or add Hebrew tokens)
- **Chunk size:** Adjust based on your model's context window
- **Mixed language:** Consider training separate models or using special language tokens