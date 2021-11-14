import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy
import random
import json
import tensorflow
import tflearn
import pickle
tf = tensorflow.compat.v1


with open("intents.json") as file:
    data = json.load(file)

newmod = input("New Model? (Y/N):")

if newmod.lower() == 'n':
    with open("data.p", "rb") as f:
        words, labels, training, output = pickle.load(f)
else:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))] # is this lables or classes

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.p", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, 12)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

if newmod.lower() == 'n':
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    user = {
        "name":"",
        "likes":[],
        "state":""
    }
    previousTag = ''

    print("Say something to get started!\nOr say \" Example Prompts \" to see examples")

    while True:
        inp = input(">> Human: ")
        if inp.lower() == 'quit':
            break
            
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if tag == "likes":
            user["likes"].append(inp)
        if tag == "tellMore" and previousTag != '':
            tag = previousTag
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        if tag == "whoami" and user["name"] != '':
            responses = ['Your name is ' + user["name"]]
        
        print(">> Chatbot:", random.choice(responses))

        if tag == "whoami" and len(user["likes"]) > 0:
            like = random.choice(user["likes"])
            like = like.replace('I am', "you are")
            like = like.replace('i am', "you are")
            like = like.replace("I'm", "you are")
            like = like.replace("i'm", "you are")
            like = like.replace('I ', "you")
            like = like.replace('i ', "you")
            
            print(">> Chatbot: You also told me", like)

        if tag == "name":
            print(">> Chatbot:", "What is your name?")
            inp = input(">> Human: ")
            user["name"] = inp
            print(">> Chatbot: Nice to meet you", inp)
        if tag == "howare":
            print(">> Chatbot:", "How about you?")
            inp = input(">> Human: ")
            user["state"] = inp
            print(">> Chatbot:", "Good to know.")
        
        previousTag = tag

chat()