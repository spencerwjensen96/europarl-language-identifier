import os
import fasttext as ft

def fasttext_predict(text):
    print(f"here - {text}")
    model = ft.load_model('/Users/spencer.jensen/Desktop/code/language_detector/training/fastText/model/europarl.bin')
    response = model.predict([text])
    print(response)
    prediction = response[0][0][0]
    confidence = response[1][0][0]
    return prediction, confidence
