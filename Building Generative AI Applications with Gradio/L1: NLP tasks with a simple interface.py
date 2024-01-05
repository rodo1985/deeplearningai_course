import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

def summarize_model(input):
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output

def summarize(input):
    output = summarize_model(input)
    return output

gr.close_all()
demo = gr.Interface(fn=summarize, 
                    inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Result", lines=3)],
                    title="Text summarization with distilbart-cnn",
                    description="Summarize any text using the `shleifer/distilbart-cnn-12-6` model under the hood!"
                   )
demo.launch()