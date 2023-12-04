import json
import pickle
import random
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def cleaning_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def group_of_words(sentence):
    sentence_words = cleaning_sentence(sentence)
    group = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                group[i] = 1
    return np.array(group)

def predict_class(sentence):
    bow = group_of_words(sentence)
    res = model.predict(np.array(bow))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
print('Chat is active!')

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
