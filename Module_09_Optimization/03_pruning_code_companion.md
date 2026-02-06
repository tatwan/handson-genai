# Model Pruning - Comprehensive Summary

---

## Overview

**Pruning removes unnecessary weights from neural networks, making them smaller and faster.**

### Learning Objectives
1. Understand different pruning strategies
2. Apply L1 unstructured pruning with PyTorch
3. Visualize sparsity in neural networks
4. Analyze the accuracy vs. sparsity trade-off

### Prerequisites
- Basic PyTorch neural network knowledge
- Understanding of histograms and distributions

---

## What is Pruning?

### The Intuition
Neural networks are often **over-parameterized**. Many weights contribute little to the output.

### Why Pruning Works: The Redundancy Hypothesis

Neural networks often exhibit significant redundancy:
- Many weights are extremely close to zero
- Some weights are correlated with others
- **The Lottery Ticket Hypothesis** suggests that dense networks contain sparse sub-networks

---

## Types of Pruning

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| **Unstructured** | Remove individual weights | Most flexible, highest sparsity | Irregular memory access |
| **Structured** | Remove entire neurons/channels | Hardware-friendly | Less flexible |
| **Magnitude-based** | Remove smallest weights | Simple, effective | May miss important small weights |
| **Gradient-based** | Remove by gradient importance | More accurate | Requires training data |

---

## Implementation

### Model Used
- **Name**: distilbert-base-uncased-finetuned-sst-2-english
- **Total Parameters**: 66,955,010
- **Task**: Sentiment Classification (SST-2)
- **Initial Sparsity**: 0.00%

### Core Pruning Function
```python
def apply_pruning(model, amount=0.3, prune_type='l1'):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if prune_type == 'l1':
                prune.l1_unstructured(module, name='weight', amount=amount)
    return model
```

---

## Experimental Results

### Weight Statistics (Before Pruning)
```
Min:   -9.9695
Max:    2.3805
Mean:  -0.0134
Std:    0.0481
```

### After 30% Pruning
```
Target Sparsity:    30.0%
Actual Sparsity:    19.29%
Total Parameters:   66,955,010
Non-zero Params:    54,037,406
```

---

## Sparsity vs. Accuracy Trade-off

| Target Sparsity | Accuracy | Accuracy Drop |
|-----------------|----------|---------------|
| 0.0%            | 90.8%    | Baseline      |
| 30.0%           | 89.6%    | -1.2%         |
| 50.0%           | 88.2%    | -2.6%         |
| 60.0%           | 83.2%    | -7.6%         |
| 80.0%           | 76.4%    | -14.4%        |

### Key Finding: **Sweet Spot = 30-50% Sparsity**
- Below 50%: Accuracy stays relatively stable
- Above 60%: Significant accuracy loss

---

## Key Insights

1. **Weight Distribution Pattern**: Most weights cluster around zero
2. **L1 Pruning Strategy**: Targets smallest absolute values
3. **Layerwise Consistency**: Uniform sparsity across all layers
4. **Accuracy Preservation**: 30-50% sparsity has minimal impact
5. **Zero Spike**: Clear histogram spike at zero after pruning

---

## Key Takeaways

1. Pruning removes weights based on importance (magnitude)
2. Unstructured pruning is flexible but may not speed up inference
3. L1 pruning removes smallest absolute values
4. 30-50% sparsity often has minimal accuracy impact
5. Trade-off exists between sparsity and accuracy

---

## Next Steps

Continue to 04_quantization.ipynb for precision reduction techniques!

