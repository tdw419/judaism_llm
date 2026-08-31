# Roadmap: Judaism LLM Development

**Goal:** Build a domain-specific LLM fine-tuned on Sefaria's corpus of Jewish texts (Torah, Talmud, Commentaries) with bilingual Hebrew-English capabilities.

**Hardware:** RTX 5090 GPU (assume 32GB+ VRAM for efficient training)

---

## Phase 1: Data Acquisition & Preparation (Current)

**Status:** Scripts created, not executed yet.

### Tasks:
- [x] Create `download_sefaria.py` script
- [x] Create `prepare_training_data.py` script
- [ ] **Run data download:**
  ```bash
  python download_sefaria.py
  ```
  - Expected: ~10,000+ JSON files organized by category
  - Time: ~30-60 mins (rate-limited API calls)
- [ ] **Prepare training data:**
  ```bash
  python prepare_training_data.py
  ```
  - Output: `training_data.jsonl` (causal LM format)
  - Output: `qa_training_data.jsonl` (instruction format)
- [ ] **Data audit:**
  - Check file counts: `find sefaria_texts/ -name "*.json" | wc -l`
  - Check JSONL line counts: `wc -l training_data.jsonl`
  - Verify Hebrew/English content samples

### Success Criteria:
- `training_data.jsonl` contains >100,000 text chunks
- Data includes both Hebrew and English
- No corrupted JSON files

---

## Phase 2: Base Model Selection

**Strategy:** Choose a strong multilingual base model with good Hebrew tokenization.

### Candidates:
1. **Qwen2.5-7B-Instruct** (Recommended)
   - Excellent multilingual support (covers Hebrew well)
   - Strong reasoning capabilities
   - 7B params fits comfortably in VRAM
2. **Llama-3.1-8B-Instruct**
   - Industry standard, easy to deploy
   - Good Hebrew support via tokenizer
3. **Mistral-Nemo-12B**
   - Larger context window (128k)
   - Strong performance on complex texts

### Decision:
- **Start with:** Qwen2.5-7B (or 14B if VRAM permits)
- **Format:** GGUF for local inference, Safetensors for training

---

## Phase 3: Fine-tuning Infrastructure

### Tasks:
- [ ] **Install training dependencies:**
  ```bash
  pip install "unsloth[cu121-torch240]" --extra-index-url https://pypi.nvidia.com
  pip install "trl>=0.9" "peft>=0.8" "bitsandbytes>=0.43"
  ```
- [ ] **Create `unsloth_finetune.py`:**
  - Load model in 4-bit quantization
  - Configure LoRA adapters (r=16, alpha=32, dropout=0.05)
  - Set training hyperparameters (learning rate, batch size)
  - Save checkpoints every 500 steps
- [ ] **Create training config:**
  - Batch size: 4 (gradient accumulation to simulate 32)
  - Learning rate: 2e-4 (with cosine decay)
  - Epochs: 3-5 (or until validation loss plateaus)
  - Max sequence length: 2048 or 4096

### Success Criteria:
- Training starts without OOM errors
- GPU utilization >80%
- Loss decreases monotonically

---

## Phase 4: Execution (Training)

### Tasks:
- [ ] **Run fine-tuning:**
  ```bash
  python unsloth_finetune.py --data training_data.jsonl --output_dir ./judaism-llm-qwen2.5-7b
  ```
- [ ] **Monitor training:**
  - Watch GPU VRAM usage
  - Track training loss vs. validation loss
  - Sample generations during training
- [ ] **Save final model:**
  - Merge LoRA adapters into base model
  - Export to GGUF for Ollama compatibility
  - Upload to Hugging Face Hub (optional)

### Success Criteria:
- Final perplexity on validation set improves over base model
- Model generates coherent Hebrew/English text
- Training completes without crashes

---

## Phase 5: Evaluation & Benchmarking

### Tasks:
- [ ] **Create test suite:**
  - Specific Torah passages (continuation test)
  - Talmudic argument structure test
  - Hebrew-English translation pairs
  - General Judaica knowledge questions
- [ ] **Run comparison:**
  - Base model responses vs. Fine-tuned model responses
  - Qualitative human evaluation
  - Quantitative metrics (BLEU for translation, perplexity)
- [ ] **Create evaluation script:**
  - `evaluate_model.py` - Automated testing pipeline

### Success Criteria:
- Fine-tuned model significantly outperforms base on domain tasks
- Model maintains general language capabilities
- Translation accuracy >80% on test set

---

## Phase 6: Deployment & Inference

### Tasks:
- [ ] **Export to Ollama:**
  ```bash
  # Convert to GGUF
  llama.cpp/quantize model.gguf model-q4_k_m.ggml q4_k_m

  # Create Ollama modelfile
  ollama create judaism-llm -f Modelfile
  ```
- [ ] **Create CLI interface:**
  - Simple chat script: `chat.py`
  - Text generation script: `generate.py`
- [ ] **Optional: Web UI:**
  - Use text-generation-webui or Oobabooga
  - Create Gradio/Streamlit interface for Q&A

### Success Criteria:
- Model loads in Ollama successfully
- Inference latency <500ms/token
- Stable operation without memory leaks

---

## Phase 7: Iteration & Expansion

### Future Enhancements:
- [ ] **RAG Integration:**
  - Add Sefaria API as retrieval source
  - Build vector index of all texts
  - Implement citation system
- [ ] **Specialized Tasks:**
  - Talmudic argument analysis
  - Hebrew grammar checking
  - Commentary cross-referencing
- [ ] **Multimodal:**
  - Add Torah scroll image OCR
  - Visual components for text layout

---

## Immediate Next Steps (Today)

1. Run `python download_sefaria.py`
2. Run `python prepare_training_data.py`
3. Decide on base model (Qwen2.5-7B vs Llama-3.1-8B)
4. Create `unsloth_finetune.py` script

---

## Project Structure

```
judaism_llm/
├── download_sefaria.py          # Phase 1 ✅
├── prepare_training_data.py     # Phase 1 ✅
├── unsloth_finetune.py          # Phase 3 (Next)
├── evaluate_model.py            # Phase 5
├── chat.py                      # Phase 6
├── Modelfile                    # Phase 6
├── sefaria_texts/               # Raw JSON downloads
├── training_data.jsonl          # Phase 1 output
├── checkpoints/                 # Phase 4 output
└── ROADMAP.md                   # This file
```