# test.py

from evaluate_speech import evaluate

word_id = "w01"          # change to target word
audio = "speech.wav"     # recorded file

result = evaluate(audio, word_id)
print(result)