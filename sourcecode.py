import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
from pprint import pprint
intents = json.loads(open('nlp.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        
    return return_list

def getResponse(sen, ints, intents_json):
    tag = ints[0]['intent']
    result=""
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            if (i['tag'] == 'greeting' or i['tag'] == 'goodbye' or i['tag'] == 'thanks'  or i['tag'] == 'noanswer' or i['tag'] == 'options' or i['tag'] == 'students_info'):
                result = random.choice(i['responses'])
                break
            elif(i['tag']== 'get_student_name'):
                lis = list(sen.split(' '))
                l= len(lis)
                j= lis[l-1]
                k="".join(j)
                sn=i['responses']
                for a in sn:
                    for b in a['students']:
                        if(b.lower() == k.lower()):
                            result =  a['students'][b]
                            break
                        else:
                            continue
                if(result==""):
                    result="name for given roll number not found"
            elif(i['tag']== 'get_student_roll'):
                lis = list(sen.split(' '))
                for idx,val in enumerate(lis):
                    if(val =='of'):
                        n=idx
                        break
                j= lis[n+1:]
                k="".join(j)
                sid=i['responses']
                for a in sid:
                    for b in a['students']:
                        if((a['students'][b].replace(' ', '').lower())==(k.replace(' ', '').lower())):
                            result = b
                            break
                        else:
                            continue
                if(result==""):
                    result=" roll number for given student name not found"
                    
            elif(i['tag'] == 'second_floor'):
                lis = list(sen.split(' '))
                print(lis)
                print()
                l= len(lis)
                for idx,val in enumerate(lis):
                    if(val =='is'):
                        n=idx
                        break
                j= lis[n+1:]
                k="".join(j)
                print(k)
                print()
                sn=i['responses']
                for a in sn:
                    for b in a['directions']:
                        if(b.replace(' ', '').lower() == k.replace(' ', '').lower()):
                            result =  a['directions'][b]
                            break
                        else:
                            continue
                if(result==""):
                    result="Location not found"

            elif(i['tag'] == 'get_labexam_date'):
                lis = list(sen.split(' '))
                l=len(lis)
                for idx,val in enumerate(lis):
                    if(val == 'of'):
                        n = idx
                        break
                j = lis[n+1:]
                k="".join(j)
                sid=i['responses']
                for a in sid:
                    for b in a['dates']:
                        if((b.replace(' ','').lower()) == (k.replace(' ','').lower())):
                            result = a['dates'][b]
                            break
                        else:
                            continue
                    if(result==""):
                        result=" please enter a valid subject"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(msg, ints, intents)
    return res

def chat():
    print("Chatbot CLI - Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if user_input != '':
            print("You: " + user_input)

            response = chatbot_response(user_input)
            print("Bot:", response)

if __name__ == "__main__":
    chat()
