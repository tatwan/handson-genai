# Hands-On Generative Artificial Intelligence

**Prepared for Intuit**
*Last Updated February 23rd, 2026*

Hands-on Generative AI is an interactive three-day training course that offers a comprehensive learning experience for developers, data engineers/analysts, and tech product owners. The course is specifically designed to equip participants with the essential skills and in-depth knowledge required to harness the power of generative AI effectively. By combining theory with extensive hands-on practice, this course ensures that participants gain a deep understanding of generative AI concepts and the ability to apply them to various domains. Students will learn how to generate realistic and novel outputs, such as images, music, text, and more, using state-of-the-art algorithms and frameworks.

## Duration
3 Days

## Prerequisites
To get the most out of this session, participants should have:
* **Python Programming:** Participants should have a solid understanding of Python programming, including knowledge of data structures, control flow, functions, and libraries commonly used in data analysis and machine learning, such as NumPy, Pandas, and scikit-learn.
* **Data Analysis and Machine Learning:** Familiarity with data analysis concepts, exploratory data analysis (EDA), and machine learning algorithms is essential.
* **Deep Learning Basics:** Basic knowledge of deep learning concepts is recommended.

## Learning objectives
After this course, participants will be able to:
* Construct and evaluate generative architectures, specifically Autoencoders and Variational Autoencoders (VAEs), to synthesize new data samples.
* Implement advanced prompting and sampling strategies (such as Top-P, Top-K, and Temperature) to control the creativity and accuracy of LLM outputs.
* Architect a Retrieval-Augmented Generation (RAG) system that integrates pre-trained models with external data to solve complex domain-specific tasks.
* Design and implement agentic AI workflows that leverage tool use, function calling, and multi-step reasoning to automate complex, goal-oriented tasks.
* Apply parameter-efficient fine-tuning techniques, including LoRA and PEFT, to adapt pre-trained models for domain-specific tasks with reduced computational cost.

## This course includes:
* **End-to-End Model Adaptation:** Comprehensive coverage of transfer learning and fine-tuning using the Hugging Face ecosystem for specific NLP tasks like sentiment analysis.
* **Hands-on Technical Labs:** A deep dive into the mathematical and programmatic foundations, including vector embeddings, backpropagation, and latent space manipulation.
* **Deployment Optimization:** Practical training on model compression techniques, including quantization, pruning, and distillation, to prepare models for real-world production.

## This course does not include:
* **Infrastructure and Cloud Provisioning:** The course does not cover the setup of GPU clusters, cloud environment administration (AWS/Azure/GCP), or MLOps pipeline orchestration.
* **Non-Transformer Architectures:** While it covers VAEs, the course does not cover GANs or their variants. While diffusion models are introduced conceptually, the course does not pursue high-fidelity image or video generation as a primary focus. The emphasis throughout remains on language models and text-based applications.
* **Ethical and Legal Frameworks:** This is a technical implementation course; it does not provide comprehensive training on AI policy, copyright law, or organizational governance structures.

---

## Outline

### Module 01: Foundations of AI and Machine Learning
* Machine Learning vs rule-based programming.
* Supervised and unsupervised learning. Examples and applications in real-world scenarios.
* An overview of ML model development and evaluation:
    * Data preprocessing
    * Feature engineering
    * Overfit
    * Model evaluation metrics
* Hands-on Lab: Training and evaluating a classifier.

### Module 02: Deep Learning Primer
* Fundamental concepts of deep learning
* Data types and volumes
* Overview of neural network structures and common architectures.
* Optimizers, gradient descent, and backpropagation algorithms.
    * Optional demo: TensorFlow playground
* Deep learning frameworks: TensorFlow and PyTorch
* Hands-on Labs: Neural network basics and Image classification using PyTorch.

### Module 03: Overview of Generative AI
* Introduction to Generative AI and its applications.
* Basic principles of generative models and their architectural components.
* Demo: a simple example of probabilistic sampling to create simulated data
* Autoencoders & Variational Autoencoders: latent space, representation learning, and sampling techniques.
* Hands-on Lab: VAE & Autoencoders: Understanding latent space and generate fake images of handwritten digits with VAE.
* Diffusion Models: Introduction and their role in modern generative AI applications.
* Hands-on Lab: Generating images using a pre-trained diffusion model pipeline from Huggingface.

### Module 04: NLP - Understanding Language as Data
* Introduction to NLP techniques and applications
    * Tokenization: BPE and WordPiece tokenization
    * Vectorization. Bag-of-Words and its limitations
    * Embeddings: mathematical text representation in a continuous vector space.
    * Modern dense embeddings
    * Similarity search
* Hands-on Lab: Find similar documents modern embedding models

### Module 05: Large Language Models (LLMs)
* NLP and text generation before the introduction of pre-trained LLMs.
* Overview of pre-trained models:
    * BERT (encoder-focused, classification tasks)
    * GPT (decoder-focused, generation tasks)
* The Transformer architecture: why it matters and how it works at a high level
* Demo: GPT as a probabilistic autoregressive model (OpenAI Playground)
* Demo: A Tour of Hugging Face (models, datasets, and spaces)
* Hands-on Lab: Introduction to BERT and GPT
* The Modern LLM Landscape
    * Open-weight models
    * Introduction to reasoning models
    * Multimodal LLMs: vision-language models and image-text reasoning
* Demo: Working with multimodal inputs using Gemini and GPT
* Working with LLM APIs and OpenAI API Compatibility
* Hands-on Lab: Building interactive UIs with Gradio

### Module 06: Prompting Techniques and Agentic AI
* Generative tasks:
    * Text completion
    * Dialogue systems
    * Summarization
    * Code generation
* Prompt refinement
    * Prompt engineering and context engineering
    * Zero-shot, few-shot, and chain-of-thought prompting
    * System prompts and instruction following
    * Structured output: JSON mode and function schemas
* Responsible Prompting: understanding prompt injection, prompt leaking, and prompt hijacking, including risks and mitigation strategies
* Hands-on Lab: Prompting techniques for summarization, code generation and text labelling
* Agentic AI
    * Tool Use and Function Calling: enabling LLMs to interact with external systems
    * Introduction to Agentic AI: multi-step reasoning and autonomous task execution
    * ReAct pattern and Agentic AI frameworks
    * Overview of the MCP (Model Context Protocol) standard for tool and context integration
* Hands-on Lab: Function calling and building simple agent using smolagents or LangGraph

### Module 07: Retrieval-Augmented Generation (RAG)
* Chunking Strategies & Embedding model selection
* Vector Databases
* Agentic RAG
* RAG Evaluation and Observability
    * Automated evaluation techniques including LLM-as-a-Judge
    * Measuring retrieval quality, answer faithfulness, and context relevance
* Hands-on Lab: Building an end-to-end RAG pipeline

### Module 08: Fine-Tuning Large Language Models
* Transfer learning and full fine-tuning strategies for LLMs.
* Considerations for cost and potential catastrophic forgetfulness.
* Using Hugging Face's transformers library for fine-tuning
* Hands-on Lab: BERT Fine tuning for sentiment analysis and summarization.
* Parameter-Efficient Fine-Tuning (PEFT)
* Hands-on Lab/Demo: Fine-tuning Small Language Model (SLM) using LoRA/QLoRA
* Sampling techniques (Temperature, Top-P, Top-K, Beam Search)
* Hands-on Lab: Customize Generative LLM Output with sampling parameters
* Hands-on Lab: Fine-Tuning with OpenAI

### Module 09: Model Optimization and Deployment
* Production challenges: memory, cost, latency
* Strategies for deploying generative models:
    * Knowledge distillation (teacher-student training)
    * Model pruning (structured and unstructured)
    * Quantization (FP16, INT8, INT4, GPTQ, AWQ)
* Benchmarking and performance evaluation
* Hands-on Labs: Knowledge distillation, Model pruning, Quantization, and Benchmarking

### Module 10: Capstone Project
* Building a production-ready dialogue system with RAG
* Integrating multiple techniques learned throughout the course
* Hands-on Lab: Capstone Dialogue System
