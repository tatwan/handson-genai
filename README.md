# Hands-On Generative Artificial Intelligence

<img src="images/Gemini_Generated_Image_7q6ku87q6ku87q6k.png" alt="Gemini_Generated_Image_7q6ku87q6ku87q6k" style="zoom:50%;" />

## Getting Started

In Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then CD into the Colab Notebooks:
```
!cd /content/drive/MyDrive/Colab\ Notebooks
```

Clone this repo:
```
!git clone https://github.com/tatwan/handson-genai.git
```

---

## Course Overview

This intensive **three-day training course** combines theory with extensive hands-on practice to teach participants how to build **production-ready generative AI applications** using state-of-the-art models and techniques.

Designed for developers, data engineers/analysts, and tech product owners, this course covers the complete GenAI lifecycle from foundations to deployment, including modern topics like diffusion models, multimodal AI, RAG systems, AI agents, and production optimization.

**Course Duration:** 3 Days  
**Last Updated:** February 6, 2026  
**Total Labs:** 27 hands-on notebooks

---

## Course Modules

### 📁 Module Structure

```
handson-genai/
├── Module_01_ML_Foundations/      # Day 1
│   └── 01_intro_to_ml_concepts.ipynb
├── Module_02_Deep_Learning/       # Day 1
│   └── 01_neural_network_basics.ipynb
├── Module_03_Generative_AI/       # Day 1
│   ├── 01_intro_to_generative_ai.ipynb
│   ├── 02_autoencoders.ipynb
│   └── 03_diffusion_models.ipynb          
├── Module_04_NLP/                 # Day 2
│   ├── 01_intro_to_nlp.ipynb
│   ├── 02_tokenization.ipynb
│   └── 03_embeddings.ipynb
├── Module_05_LLMs/                # Day 2
│   ├── 01_openai_ollama.ipynb
│   ├── 02_huggingface_tour.ipynb
│   ├── 03_bert_gpt.ipynb
│   ├── 04_gradio.ipynb
│   └── 05_multimodal_models.ipynb         
├── Module_06_Prompting/           # Day 2
│   ├── 01_prompting_techniques.ipynb
│   ├── 02_function_calling.ipynb
│   ├── 03_langchain_function_calling.ipynb
│   └── 04_react_agent.ipynb
├── Module_07_Fine_Tuning/         # Day 3
│   ├── 01_transfer_learning.ipynb
│   ├── 02_sentiment_analysis.ipynb        
│   ├── 03_fine_tuning_openai.ipynb
│   ├── 04_Fine_Tuning_LLM_Healthcare.ipynb
│   └── 05_sampling_techniques.ipynb
├── Module_08_RAG/                 # Day 3
│   ├── 01_rag_langchain.ipynb
│   ├── 02_rag_llamaindex.ipynb
│   └── 03_rag_evaluation.ipynb
├── Module_09_Optimization/        # Day 3 
│   ├── 01_intro_to_optimization.ipynb
│   ├── 02_knowledge_distillation.ipynb
│   ├── 03_pruning.ipynb
│   └── 04_quantization.ipynb
└── Module_10_Capstone/            # Day 3
    └── capstone_dialogue_system.ipynb
```

---

## Day-by-Day Outline

### Day 1: Foundations and Generative AI

#### Module 1: Foundations of AI and Machine Learning
- Machine Learning vs rule-based programming
- Supervised and unsupervised learning with examples
- ML model development workflow: preprocessing, features, overfitting, evaluation
- **Lab:** `01_intro_to_ml_concepts.ipynb`

#### Module 2: Deep Learning Primer
- Fundamental concepts of neural networks
- Optimizers, gradient descent, and backpropagation
- Deep learning frameworks: TensorFlow and PyTorch
- **Lab:** `01_neural_network_basics.ipynb`

#### Module 3: Overview of Generative AI
- Introduction to Generative AI and applications
- Probabilistic sampling and latent space concepts
- Autoencoders and Variational Autoencoders (VAEs)
- **Diffusion models for image generation** 
- **Labs:** `01_intro_to_generative_ai.ipynb`, `02_autoencoders.ipynb`, `03_diffusion_models.ipynb`

---

### Day 2: NLP, LLMs, and Intelligent Agents

#### Module 4: NLP - Understanding Language as Data
- Tokenization and text preprocessing
- Vectorization and embeddings (Word2vec)
- **Labs:** `01_intro_to_nlp.ipynb`, `02_tokenization.ipynb`, `03_embeddings.ipynb`

#### Module 5: Large Language Models (LLMs)
- Pre-trained models: BERT and GPT
- Working with OpenAI and Ollama APIs
- Hugging Face ecosystem tour
- Building UIs with Gradio
- **Multimodal AI: Vision-language models (GPT-4V)** 
- **Labs:** `01_openai_ollama.ipynb`, `02_huggingface_tour.ipynb`, `03_bert_gpt.ipynb`, `04_gradio.ipynb`, `05_multimodal_models.ipynb`

#### Module 6: Prompting Techniques and Agentic AI
- Zero-shot, few-shot, and chain-of-thought prompting
- Function calling and tool use (OpenAI and LangChain)
- **ReAct agents for autonomous workflows**
- **Labs:** `01_prompting_techniques.ipynb`, `02_function_calling.ipynb`, `03_langchain_function_calling.ipynb`, `04_react_agent.ipynb`

---

### Day 3: Fine-Tuning, RAG, Optimization & Capstone

#### Module 7: Fine-Tuning Large Language Models
- Transfer learning and fine-tuning strategies
- **LoRA and Parameter-Efficient Fine-Tuning (PEFT)**
- **Sentiment analysis with DistilBERT** 
- Catastrophic forgetting prevention
- Sampling techniques: Temperature, Top-P, Top-K
- **Healthcare LLM fine-tuning**
- **Labs:** `01_transfer_learning.ipynb`, `02_sentiment_analysis.ipynb`, `03_fine_tuning_openai.ipynb`, `04_Fine_Tuning_LLM_Healthcare.ipynb`, `05_sampling_techniques.ipynb`

#### Module 8: Retrieval-Augmented Generation (RAG)
- RAG architecture with LangChain and LlamaIndex
- **Vector databases (Chroma) and semantic search**
- **RAG evaluation and observability with MLflow**
- **Labs:** `01_rag_langchain.ipynb`, `02_rag_llamaindex.ipynb`, `03_rag_evaluation.ipynb`

#### Module 9: Model Optimization and Deployment 
- Production challenges: memory, cost, latency
- **Knowledge distillation** (teacher-student training)
- **Model pruning** (structured and unstructured)
- **Quantization** (FP16, INT8, INT4, GPTQ, AWQ)
- Deployment strategies for production
- **Labs:** `01_intro_to_optimization.ipynb`, `02_knowledge_distillation.ipynb`, `03_pruning.ipynb`, `04_quantization.ipynb`

#### Module 10: Capstone Project
- Build a complete RAG-based dialogue system
- Integrate multiple techniques learned throughout the course
- **Lab:** `capstone_dialogue_system.ipynb`

---

## What You'll Learn

By the end of this course, you will be able to:

✅ **Build applications with modern LLMs** (GPT-4, Claude, Llama, Mistral)  
✅ **Generate images** with diffusion models (Stable Diffusion)  
✅ **Create multimodal applications** using vision-language models (GPT-4V)  
✅ **Fine-tune models efficiently** using LoRA and PEFT techniques  
✅ **Implement RAG systems** with vector databases (Chroma)  
✅ **Create AI agents** with function calling and ReAct patterns  
✅ **Optimize models for production** (distillation, pruning, quantization)  
✅ **Deploy GenAI applications** with proper evaluation and monitoring  
✅ **Use industry-standard tools** (Hugging Face, LangChain, LlamaIndex, MLflow)

---

## Prerequisites

- **Python Programming:** Solid understanding of Python, including data structures, control flow, functions, and libraries like NumPy and Pandas
- **Machine Learning Fundamentals:** Familiarity with supervised/unsupervised learning, model evaluation, and scikit-learn
- **Deep Learning Basics:** Understanding of neural networks recommended but not required
- **API Experience:** Helpful to have worked with REST APIs (not required)

---

## Course Features

• **27 Hands-on Labs:** Practical notebooks covering every major topic  
• **Production-Focused:** Learn deployment and optimization techniques  
• **Modern Tools:** Work with 2026 industry-standard frameworks  
• **Complete Lifecycle:** From model selection to production deployment  
• **Real-World Projects:** Build a complete RAG-based dialogue system

---

## Archived Content

Previous demos and labs are available in the `_archive/` folder for reference.
