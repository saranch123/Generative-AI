from transformers import pipeline
import gradio as gr

# Load the sentiment-analysis pipeline
print("Loading model...")
classifier = pipeline("sentiment-analysis")
print("Model loaded.")

def analyze_text(text):
    result = classifier(text)[0]
    return f"Label: {result['label']}, Score: {round(result['score'], 4)}"

demo = gr.Interface(
    fn=analyze_text,
    inputs="text",
    outputs="text",
    title="My First AI Sentiment App",
    description="Type a sentence below to see if it's Positive or Negative!"
)

if __name__ == "__main__":
    print("Launching Gradio app...")
    demo.launch(share=True)
