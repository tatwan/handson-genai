# Text Summarization with DialogSum

This notebook introduces **sequence-to-sequence (seq2seq) models** for abstractive summarization, where the model generates new text rather than selecting existing sentences.

------

## Why Seq2Seq Models?

## Architecture Comparison

Understanding **why** we use different architectures for different tasks:

| Architecture                   | How It Works                             | Best For                   | Example            |
| :----------------------------- | :--------------------------------------- | :------------------------- | :----------------- |
| **Encoder-only** (BERT)        | Reads text, outputs embeddings           | Classification, NER        | Sentiment analysis |
| **Decoder-only** (GPT)         | Generates text left-to-right             | Text generation, chat      | Story completion   |
| **Encoder-Decoder** (T5, BART) | Reads input fully, then generates output | Translation, summarization | Dialogue → Summary |

**Why T5 for summarization?**
The encoder reads the **entire dialogue context** before the decoder starts generating. This prevents the model from "forgetting" early parts of long conversations, unlike decoder-only models that process sequentially.​

```
Input: "Person A said X. Person B replied Y. Person A then said Z."
         ↓ ENCODER (understands full context)
         ↓ DECODER (generates summary considering all of A, B, C)
Output: "Person A and B discussed X, resulting in Z."
```

------

## T5's Task Prefix System

## Why the "summarize:" Prefix?

T5 was pre-trained on multiple tasks simultaneously. The prefix tells the model **which task mode**to activate:

```python
prefix = "summarize: "
inputs = [prefix + doc for doc in examples['dialogue']]
```

**Other T5 prefixes**:

- `"translate English to French: "`
- `"question: ... context: ..."`
- `"cola sentence: "` (grammar checking)

This design allows **one model** to handle multiple tasks without architecture changes.

------

## Understanding ROUGE Metrics

## What is ROUGE?

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) measures **n-gram overlap** between generated and reference summaries.

## Visual Example

```
Reference:  "The cat is on the mat"
Generated:  "The cat is on"

ROUGE-1 (Unigrams):
  Matches: "The", "cat", "is", "on" → 4 out of 6 words
  Score: 4/6 = 67%

ROUGE-2 (Bigrams):
  Reference pairs: ["The cat", "cat is", "is on", "on the", "the mat"]
  Generated pairs: ["The cat", "cat is", "is on"]
  Matches: 3 out of 5 → Score: 60%

ROUGE-L (Longest Common Subsequence):
  Longest matching sequence: "The cat is on" (4 words)
  Measures structural similarity
```

## Interpreting ROUGE Scores

ROUGE scores are **percentages** of overlap, not quality ratings:

| Score Range         | Interpretation                                 |
| :------------------ | :--------------------------------------------- |
| **ROUGE-1: 35-45%** | Good vocabulary overlap; captures key entities |
| **ROUGE-2: 15-25%** | Reasonable phrase-level similarity             |
| **ROUGE-L: 30-40%** | Strong structural alignment                    |

**Critical limitation**: A semantically perfect summary using **synonyms** will score low:

```
Reference:  "The CEO resigned due to financial losses"
Generated:  "The chief executive quit because of monetary deficit"
ROUGE-1: ~0% (no exact word matches!)
Human evaluation: Perfect summary ✓
```

**Takeaway**: Use ROUGE for **relative comparison** during training, but validate final quality with human evaluation.

------

## Critical Technical Detail: Token Decoding Bug

## Why the Complex `compute_metrics` Function?

The notebook includes extensive token cleaning code that might look unnecessary. Here's **why it's critical**:

```python
# This will FAIL with OverflowError on some systems:
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

# Why? Three reasons:
# 1. Training labels use -100 for padding (HuggingFace convention)
# 2. NumPy int64 values don't convert cleanly to Rust integers (Fast Tokenizer)
# 3. Negative values are invalid token IDs
```

**The fix**:

```
python
predictions_clean = []
for row in predictions:
    clean_row = []
    for tok in row:
        tok_int = int(tok)  # Convert numpy.int64 → Python int
        if tok_int < 0 or tok_int >= vocab_size:
            clean_row.append(pad_id)  # Replace invalid tokens
        else:
            clean_row.append(tok_int)
    predictions_clean.append(clean_row)
```

**When you'll encounter this**: Only when using `Seq2SeqTrainer` with `predict_with_generate=True`and HuggingFace Fast Tokenizers. Standard classification tasks don't have this issue because they don't generate text.

------

## Generation Strategies: When to Use What

## Decoding Methods Compared

The notebook demonstrates three strategies. Here's **when to use each**:

## 1. Greedy Decoding (Default)

```python
summarizer(text, do_sample=False)  # Always picks highest probability word
```

**Use when**: Speed matters, deterministic output needed (APIs, production)
**Avoid when**: Output feels repetitive or boring

## 2. Beam Search

```python
summarizer(text, num_beams=4, do_sample=False)  # Keeps top 4 paths
```

**Use when**: Quality > speed, willing to wait 2-4x longer
**Avoid when**: Running on CPU or need real-time responses

**How it works**: At each step, instead of picking the single best word, keep the top `num_beams`candidates and explore all paths:

```
Step 1: "The" → Keep top 4: ["The cat", "The dog", "The man", "The woman"]
Step 2: Expand each → Keep best 4 complete sequences
Result: Usually more coherent than greedy
```

## 3. Sampling (Creative)

```python
summarizer(text, do_sample=True, temperature=0.7, top_k=50)
```

**Use when**: Want variety (creative writing, multiple summary options)
**Avoid when**: Need factual accuracy (medical, legal, financial domains)

**Parameters**:

- `temperature=0.1`: Safe, conservative (almost like greedy)
- `temperature=0.7`: Balanced creativity
- `temperature=1.5`: Very creative (risk of hallucination)

------

## Memory Requirements & Optimization

## Why T5 Uses More Memory Than BERT

**T5-small** (60M params) uses **more GPU memory** than **DistilBERT** (66M params) during training. Why?

1. **Two models in one**: Encoder (processes input) + Decoder (generates output)
2. **Generation requires caching**: Decoder must store all previous tokens' hidden states
3. **Seq2Seq training**: Both input AND output sequences are in memory simultaneously

**Typical memory usage**:

```
DistilBERT (classification): 4-6 GB GPU
T5-small (summarization):    6-8 GB GPU
T5-base (220M params):       12-16 GB GPU (requires A100 or multiple GPUs)
```

## OOM Solutions (In Order of Effectiveness)

```python
# 1. Reduce batch size (most effective)
per_device_train_batch_size=8  # → 4 → 2

# 2. Reduce sequence lengths
max_input_length = 512   # → 256
max_target_length = 128  # → 64

# 3. Use gradient accumulation (simulates larger batches)
gradient_accumulation_steps=2  # Effective batch = 2 × 2 = 4

# 4. Reduce training data size
train_size = 1000  # → 500
```

------

## Preprocessing: The Critical Difference

## Why Seq2Seq Preprocessing is Different

Classification tasks tokenize **inputs only**. Seq2seq must tokenize **both inputs AND targets**:

```python
def preprocess_function(examples):
    # 1. Add task prefix to inputs
    inputs = [prefix + doc for doc in examples['dialogue']]
    
    # 2. Tokenize inputs (source text)
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=True
    )
    
    # 3. Tokenize targets (summaries) - THIS IS NEW
    labels = tokenizer(
        examples['summary'],  # Target summaries
        max_length=max_target_length,
        truncation=True,
        padding=True
    )
    
    # 4. Add labels to model inputs
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs
```

**Why `labels` matter**: During training, the model compares its generated tokens against `labels` to compute loss. Without properly formatted labels, the model can't learn.

------

## Compression Ratio Analysis

## Why Check Compression Ratio?

The notebook computes this statistic:

```python
ratios = [len(summary)/len(dialogue) for summary, dialogue in zip(...)]
print(f"Compression ratio: {np.mean(ratios):.1%}")
# Typical output: ~25% (summaries are 1/4 the length of dialogues)
```

**Why this matters**:

- **Too high (>50%)**: Summaries might be extractive (copying sentences) rather than abstractive
- **Too low (<10%)**: Model might be over-compressing and losing information
- **Balanced (20-30%)**: Good abstractive summarization

**Use case**: If your domain requires different compression (e.g., legal briefs need 5%, meeting notes need 40%), adjust `max_target_length` accordingly.