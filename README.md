---
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
