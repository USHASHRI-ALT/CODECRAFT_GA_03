# CODECRAFT Task 3 – Text Summarization

This project demonstrates how to perform text summarization using a Transformer-based model from HuggingFace. The script reads a text file and outputs a summarized version.

## Files Included
- `text_summarization.py`: Main script to perform summarization.
- `summarize_model.py`: Loads and prepares the summarization model.
- `example_input.txt`: Sample text to test summarization.
- `requirements.txt`: Python dependencies.

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. It has applications across many industries and is becoming an essential part of modern technology.

transformers
torch
sentencepiece

from transformers import pipeline

def load_summarizer():
    return pipeline("summarization")

from summarize_model import load_summarizer

def read_text(file_path):
    with open(file_path, "r") as file:
        return file.read()

def write_summary(summary, output_file="summary_output.txt"):
    with open(output_file, "w") as file:
        file.write(summary)

if __name__ == "__main__":
    summarizer = load_summarizer()
    text = read_text("example_input.txt")
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    write_summary(summary)
    print("Summary written to summary_output.txt")

