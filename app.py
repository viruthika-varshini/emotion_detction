from transformers import pipeline

model_checkpoint = "MuntasirHossain/RoBERTa-base-finetuned-emotion"

model = pipeline("text-classification", model=model_checkpoint)


def classify(text):
    label = model(text)[0]["label"]
    return label

description = "This AI model is trained to classify texts expressing human emotion into six categories: sadness, joy, love, anger, fear, and surprise." 
title = "Classify Texts Expressing Emotion"
# theme = "peach"
examples=[["This is such a beautiful place"]]

gr.Interface(fn=classify,
    inputs="textbox",
    outputs="text",
    title=title,
    # theme = theme,
    description=description,
    examples=examples,
).launch()
