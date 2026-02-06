# Model Quantization Explained

This document explains **what quantization is**, **how it works**, and **when to use it** for optimizing deep learning models. It serves as a companion guide to the quantization notebook.

------

## What is Quantization?

Quantization reduces the numerical precision of model weights and activations, significantly decreasing memory usage and improving inference speed.

## Numerical Precision Formats

Different data types use different amounts of memory:

| Format   | Bits | Bytes | Range       | Typical Use                        |
| :------- | :--- | :---- | :---------- | :--------------------------------- |
| **FP32** | 32   | 4     | ±3.4×10³⁸   | Training (default)                 |
| **FP16** | 16   | 2     | ±65,504     | Mixed precision training           |
| **BF16** | 16   | 2     | ±3.4×10³⁸   | Same range as FP32, less precision |
| **INT8** | 8    | 1     | -128 to 127 | Inference                          |
| **INT4** | 4    | 0.5   | -8 to 7     | Aggressive compression             |

**Key insight**: A 100M parameter model in FP32 takes 400MB, but only 100MB in INT8—a 75% reduction.

------

## Why Quantization Works

Deep learning models are remarkably robust to noise. Three key reasons explain why:

## 1. Weight Distributions Are Clustered

Neural network weights typically follow a **normal (Gaussian) distribution** centered around zero. Most values are small, so we don't need the full range of FP32.

## 2. Neural Networks Have Built-in Redundancy

Deep networks are **over-parameterized**—they have more parameters than strictly necessary. This redundancy means small errors from quantization are absorbed without affecting outputs.

## 3. Quantization Noise Acts Like Regularization

Small perturbations to weights can actually help generalization, similar to techniques like dropout.

**Analogy**: Think of image compression (JPEG). You can reduce file size by 90% while keeping the image visually identical. Quantization does the same for neural network weights.

------

## Types of Quantization

## Dynamic Quantization (INT8)

**How it works**: Quantizes weights to INT8 while keeping activations in FP32. Scales are computed on-the-fly during inference.

**Best for**:

- CPU inference
- RNN/LSTM models
- Production deployment on CPU-only servers

**Advantages**:

- 65-75% memory savings
- 2-4x speedup on CPU
- No calibration data needed
- Easy to implement

**Code example** (from notebook):

```python
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8   # Target precision
)
```

**Typical accuracy loss**: ~1%

------

## Half Precision (FP16/BF16)

**How it works**: Converts all parameters from 32-bit to 16-bit floats.

**Best for**:

- GPU inference
- Minimal accuracy loss requirements
- General GPU deployment

**Advantages**:

- 50% memory savings
- 1.5-2x speedup on GPU
- Negligible accuracy loss
- One line of code: `model.half()`

**Code example**:

```python
model_fp16 = model_fp32.half().to(device)
```

**When to use**: This should be your **first optimization** for GPU inference.

------

## 4-bit Quantization with BitsAndBytes

**How it works**: Uses specialized quantization schemes like **NF4 (NormalFloat4)**, designed for normally-distributed weights.

## What is NF4?

NF4 is a data type specifically designed for weights that follow a Normal (Gaussian) distribution. Its quantization bins are spaced based on **quantiles** of the Normal distribution, making it information-theoretically optimal for holding normally distributed data.

**Best for**:

- Very large LLMs (7B+ parameters)
- Consumer GPUs (RTX 3090, 4090)
- Memory-constrained environments

**Advantages**:

- 87.5% memory savings
- Enables running 7B models on consumer hardware
- Double quantization for even more compression

**Code example**:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # Nested quantization
)

model_4bit = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Typical accuracy loss**: 2-5%

------

## Production Quantization: GPTQ and AWQ

For production LLM serving, **GPTQ** and **AWQ** offer the best performance.

## GPTQ (GPT Quantization)

GPTQ is a **one-shot weight quantization method** based on approximate second-order information. It quantizes each layer independently while minimizing reconstruction error.

**Key features**:

- Pre-quantized models (quantization happens once, not at runtime)
- 75-87.5% memory savings
- 2-4x inference speedup
- Minimal accuracy loss (1-3%)

**Best for**: High-throughput production API serving

## AWQ (Activation-aware Weight Quantization)

AWQ improves upon GPTQ by considering the **importance of weights based on activation magnitudes**. It protects the most important weights while aggressively quantizing less important ones.

**Key features**:

- Better accuracy retention than GPTQ
- 75-87.5% memory savings
- 2-4x inference speedup
- ~2% accuracy loss

**Best for**: Accuracy-critical applications, edge deployment

## Comparison Table

| Aspect                | BitsAndBytes      | GPTQ               | AWQ                    |
| :-------------------- | :---------------- | :----------------- | :--------------------- |
| **Quantization Time** | On-the-fly        | Pre-quantized      | Pre-quantized          |
| **Inference Speed**   | Moderate          | Fast               | Fast                   |
| **Accuracy (4-bit)**  | Good              | Better             | Best                   |
| **Ease of Use**       | Very easy         | Moderate           | Moderate               |
| **Best For**          | Quick experiments | Production serving | Accuracy-critical apps |

------

## Decision Framework: Which Quantization Method to Use?

Follow this decision tree:

```
START: What hardware are you deploying to?

├─ CPU Only?
│  └─ Use Dynamic INT8 Quantization
│     • 65% memory savings
│     • 2-4x speedup on CPU
│     • No GPU required
│
└─ GPU Available?
   ├─ Model fits in memory with FP16?
   │  └─ Use FP16/BF16
   │     • 50% memory savings
   │     • Minimal accuracy loss
   │
   └─ Model too large for FP16?
      ├─ Need maximum accuracy?
      │  └─ Use 8-bit BitsAndBytes or GPTQ/AWQ
      │     • 75% memory savings
      │     • 1-2% accuracy loss
      │
      └─ Need maximum compression?
         └─ Use 4-bit NF4 or GPTQ/AWQ
            • 87.5% memory savings
            • 2-5% accuracy loss
```

## Key Selection Criteria

1. **GPU Memory**: How much VRAM do you have?
2. **Latency Requirements**: How fast do you need responses?
3. **Accuracy Tolerance**: How much quality loss is acceptable?
4. **Model Size**: How many parameters?

------

## When Quantization Fails

Quantization isn't always the right choice. Here are scenarios where it can significantly degrade performance:

## 1. Small Models (<100M parameters)

These models have **less redundancy**, so every parameter matters. Quantization errors compound and cause significant accuracy loss.

## 2. High-Precision Tasks

Applications requiring exact numerical outputs (scientific computing, financial calculations) may not tolerate quantization noise.

## 3. Embedding Layers and Output Heads

These layers are often more **sensitive to quantization**. Some frameworks leave these in higher precision.

## 4. Fine-tuning After Quantization

Quantized models are harder to fine-tune. It's better to **quantize after fine-tuning**, not before.

## Rule of Thumb

**The larger the model, the better it handles quantization**.

| Model Size     | Accuracy Drop (4-bit) | Recommendation                |
| :------------- | :-------------------- | :---------------------------- |
| 100M params    | 5-15%                 | Avoid 4-bit, use FP16         |
| 100M-1B params | 2-5%                  | Use 8-bit instead             |
| 1B-10B params  | 1-3%                  | 4-bit works well              |
| 10B+ params    | 1-2%                  | Excellent candidate for 4-bit |

------

## Memory and Speed Comparison

## Memory Usage (100M Parameter Model)

| Format              | Size   | Reduction |
| :------------------ | :----- | :-------- |
| FP32                | 400 MB | 0%        |
| FP16/BF16           | 200 MB | 50%       |
| INT8                | 150 MB | 65%       |
| INT8 (BitsAndBytes) | 100 MB | 75%       |
| INT4                | 50 MB  | 87.5%     |

## Speed Improvement

| Method       | CPU  | GPU    |
| :----------- | :--- | :----- |
| FP32         | 1x   | 1x     |
| FP16         | -    | 1.5-2x |
| INT8 Dynamic | 2-4x | -      |
| 8-bit BnB    | -    | 1-2x   |
| GPTQ/AWQ     | 2-4x | 2-4x   |

------

## Code Examples from Notebook

## 1. Dynamic INT8 Quantization

```python
# Apply dynamic quantization
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8   # Target precision
)

# Compare sizes
fp32_size = get_model_size(model_fp32)  # 255.4 MB
int8_size = get_model_size(model_int8)  # 91.0 MB
# Reduction: 64.4%
```

## 2. FP16 Conversion

```python
# Convert to FP16 (one line!)
model_fp16 = model_fp32.half().to(device)

# Memory saved: 50%
```

## 3. 4-bit BitsAndBytes

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model_4bit = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

------

## Calibration Data (for Static Quantization)

Some quantization methods (static quantization, GPTQ) require **calibration data** to:

- Determine the range of activation values
- Set optimal scale factors for quantization
- Minimize accuracy loss by observing real data distributions

**Note**: Dynamic quantization (what we use in the notebook) doesn't require calibration—it computes scales on-the-fly.

## Calibration Best Practices

1. Use **100-1000 representative samples** from your target domain
2. Include diverse examples (different lengths, topics, edge cases)
3. Don't use training data—use validation or holdout data

------

## Key Takeaways

1. **Quantization is essential** for deploying large models on constrained hardware
2. **FP16/BF16** is the easiest optimization with minimal accuracy loss
3. **Dynamic INT8** works well for CPU inference
4. **4-bit quantization** enables large models on consumer hardware
5. **Trade-off exists** between size reduction and accuracy
6. **Calibration data matters** for static quantization—use 100-1000 representative samples
7. **GPTQ and AWQ** are the go-to methods for production LLM deployment
8. **Larger models tolerate quantization better** than smaller ones