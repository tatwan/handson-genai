# Fine-Tuning Research Notes

## Parameter-Efficient Fine-Tuning (PEFT)

### Key Concepts

**What is PEFT?**
- PEFT approaches only fine-tune a small number of (extra) model parameters while freezing most parameters of the pretrained LLMs
- Greatly decreases computational and storage costs
- Overcomes catastrophic forgetting observed during full fine-tuning
- Better performance in low-data regimes and generalizes better to out-of-domain scenarios

**Benefits:**
- Tiny checkpoints (few MBs) vs full fine-tuning (GBs)
- Same base LLM can be used for multiple tasks by adding small weights
- Enables fine-tuning on consumer hardware (11GB RAM)
- Performance comparable to full fine-tuning

### PEFT Methods Supported by Hugging Face

1. **LoRA (Low-Rank Adaptation)**
   - Most popular method
   - Creates low-rank matrices for weight updates
   - Typical parameters: r=8, lora_alpha=32, lora_dropout=0.1
   - Only 0.19% of parameters are trainable

2. **Prefix Tuning**
3. **Prompt Tuning**
4. **P-Tuning**

### QLoRA
- Combines LoRA with quantization
- Uses 4-bit quantization of base model
- 4x reduction in memory usage compared to standard LoRA
- Enables fine-tuning of massive LLMs on consumer hardware

## Implementation with Hugging Face PEFT

### Basic Steps:

```python
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType

# 1. Load base model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

# 2. Create PEFT config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

# 3. Wrap model with PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# Output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19

# 4. Train normally, then save
model.save_pretrained("output_dir")

# 5. Load for inference
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
```

## Medical/Healthcare Datasets on Hugging Face

### Available Datasets:
1. **MedQuAD** - Medical Question Answering Dataset
   - keivalya/MedQuad-MedicalQnADataset
   - Question-answer pairs from 12 trusted NIH sources

2. **MedMCQA** - Multiple Choice Medical QA
   - openlifescienceai/medmcqa
   - Real-world medical entrance exam questions

3. **MedQA** - Medical Board Exam Questions
   - bigbio/med_qa
   - Professional medical board exams (US, China, Taiwan)

4. **Medical QA Collections**
   - lavita/medical-qa-datasets (collection)
   - Malikeh1375/medical-question-answering-datasets

## Ollama Integration

### Key Points:
- Ollama is for running models locally
- Fine-tuning is typically done with Unsloth or Hugging Face
- After fine-tuning, models can be exported to GGUF format for Ollama
- Workflow: Fine-tune → Export to GGUF → Import to Ollama

### Recommended Tools:
- **Unsloth**: Fast fine-tuning library (2x faster than standard methods)
- **Hugging Face Transformers + PEFT**: Standard approach
- **bitsandbytes**: For quantization and INT8 training

## Smaller Models for Demonstration

Best options for educational purposes:
1. **TinyLlama** (1.1B parameters) - Very fast
2. **Phi-2** (2.7B parameters) - High quality, small size
3. **Qwen 2.5** (0.5B-3B variants) - Efficient and capable
4. **Gemma 2B** - Google's small model


## MedQuAD Dataset Details

**Dataset:** keivalya/MedQuad-MedicalQnADataset

**Statistics:**
- Total rows: 16,407 question-answer pairs
- Size: 22.5 MB (8.82 MB in Parquet format)
- Split: train only
- Task: Question Answering

**Structure:**
- `qtype`: Type of question (string) - categories like "susceptibility", "symptoms", "treatment", "prevention", "exams and tests"
- `Question`: The medical question (string)
- `Answer`: The answer to the question (string)

**Example Questions:**
- "Who is at risk for Lymphocytic Choriomeningitis (LCM)?"
- "What are the symptoms of Lymphocytic Choriomeningitis?"
- "How to diagnose Lymphocytic Choriomeningitis (LCM)?"
- "What are the treatments for Lymphocytic Choriomeningitis?"
- "How to prevent Lymphocytic Choriomeningitis (LCM)?"

**Loading the dataset:**
```python
from datasets import load_dataset
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
```

**Reference:**
- "A Question-Entailment Approach to Question Answering". Asma Ben Abacha and Dina Demner-Fushman. BMC Bioinformatics, 2019.
