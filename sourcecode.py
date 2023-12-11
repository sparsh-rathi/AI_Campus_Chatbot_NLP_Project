from pprint import pprint
import random
import json
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
from tkinter import Tk, Scrollbar, Text, Entry, Button, END, Label
from tkinter import ttk

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('database.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.0
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(sen, ints, intents_json):
    tag = ints[0]['intent']
    result = ""
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if (i['tag'] == tag):
            if (i['tag'] == 'greeting' or i['tag'] == 'goodbye' or i['tag'] == 'thanks' or i['tag'] == 'noanswer' or i['tag'] == 'options' or i['tag'] == 'students_info'):
                result = random.choice(i['responses'])
                break
            if i['tag'] == 'directions':
                if any(keyword in sen.lower() for keyword in ["where is", "find", "navigate to", "where can i find"]):
                    location = None
                    for pattern in i['patterns']:
                        if pattern.lower() in sen.lower():
                            for floor, locations in i['responses'][0].items():
                                if any(loc.lower() in sen.lower() for loc in locations.keys()):
                                    location = next(
                                        loc for loc in locations if loc.lower() in sen.lower())
                                    break
                    if floor:
                        result = f"The {location} is located at: {i['responses'][0][floor][location]}"
                    else:
                        result = "I'm sorry, I couldn't determine the location you're asking about."
                    break
                else:
                    result = "I'm sorry, I didn't understand the location-related question."
                break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)

    if not ints:
        print("No intent predicted.")
        return "I'm sorry, I didn't understand that."

    tag = ints[0]['intent']
    res = getResponse(msg, ints, intents)
    return res


class ChatGUI:
    def __init__(self, master):
        self.master = master
        master.title("Campus Bots")
        
        master.configure(bg='#d5bdaf')

        self.chat_history = Text(
            master,
            wrap="word",
            state="disabled",
            height=20,
            width=50,
            bg='#E3D5CA', 
            relief="flat", 
            font=("Calibri", 12), 
        )
        self.chat_history.pack(expand=True, fill="both", padx=10, pady=10)  

        self.input_entry = Entry(master, width=50)
        self.input_entry.bind("<Return>", self.send_message)
        self.input_entry.pack(side="left", expand=True, fill="x", padx=10, pady=10)  

        send_button = Button(master, text="Send", command=self.send_message)
        send_button.pack(side="right", padx=10, pady=10)  

    def send_message(self, event=None):
        user_input = self.input_entry.get()
        self.input_entry.delete(0, END)

        if user_input.lower() == "exit":
            self.master.destroy()
            return

        response = chatbot_response(user_input)
        self.update_chat_history(f"You: {user_input}\nBot: {response}\n")

    def update_chat_history(self, message):
        self.chat_history.config(state="normal")
        self.chat_history.insert("end", message)
        self.chat_history.config(state="disabled")
        self.chat_history.see("end")


if __name__ == "__main__":
    root = Tk()
    root.geometry("600x500+200+100")
    style = ttk.Style(root)
    style.configure("TButton", padding=6, relief="flat", background="#4CAF50", foreground="#FFFFFF") 

    gui = ChatGUI(root)
    root.mainloop()
