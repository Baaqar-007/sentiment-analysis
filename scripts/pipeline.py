from transformers import pipeline 
import pyperclip as ppc

sent_pipeline = pipeline("sentiment-analysis")

print("Analysid of the copied data:", sent_pipeline(ppc.paste()))