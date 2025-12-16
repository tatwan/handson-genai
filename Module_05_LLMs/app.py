import gradio as gr
from transformers import pipeline
import torch

# Load models
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", device=device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

def analyze_sentiment(text):
    if not text.strip():
        return {"Error": 1.0}
    result = classifier(text)[0]
    return {
        f"{result['label']} {'😊' if result['label'] == 'POSITIVE' else '😠'}": result['score'],
        "Other": 1 - result['score']
    }

def summarize(text, max_len, min_len):
    if len(text) < 50:
        return "Please enter at least 50 characters."
    result = summarizer(text, max_length=max_len, min_length=min_len)
    return result[0]['summary_text']

with gr.Blocks(title="NLP Toolkit") as app:
    gr.Markdown("# 🛠️ NLP Toolkit")
    
    with gr.Tabs():
        with gr.TabItem("Sentiment"):
            gr.Interface(
                fn=analyze_sentiment,
                inputs=gr.Textbox(lines=3),
                outputs=gr.Label(),
                examples=[["I love this!"], ["This is terrible."]]
            )
        
        with gr.TabItem("Summarize"):
            gr.Interface(
                fn=summarize,
                inputs=[
                    gr.Textbox(lines=6),
                    gr.Slider(50, 200, value=100),
                    gr.Slider(20, 80, value=30)
                ],
                outputs=gr.Textbox()
            )

app.launch()