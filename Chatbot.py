import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import random

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

sentinel = "quit"

user_model = {"name": "Human"}


def main():

    df = pd.read_csv("knowledgebase.csv")
    df.drop_duplicates()

    userInput = "boog"
    while not userInput == sentinel:
        userInput = input(">>" + user_model["name"] + ": ")
        if not userInput == sentinel:
            tokenized_input = process_input(userInput)
            print(tokenized_input)
            matches = find_matching_term(tokenized_input, df)
            resp_idx = select_response(matches)
            chat_reply(resp_idx, df)


def process_input(input):

    tokenized_input = word_tokenize(input)
    for i in range(len(tokenized_input)):
        w = tokenized_input[i]
        w = w.lower()
        # lemmed = lemmatizer.lemmatize(w)
        tokenized_input[i] = w
    filtered = [w for w in tokenized_input if not w.lower() in stop_words]
    return filtered


def find_matching_term(tokens, df):
    matching = {}
    for index, value in df.word.items():
        if value in tokens:
            matching[index] = value
    return matching


def select_response(matches):
    if len(list(matches.items())) > 0:
        idx, response = random.choice(list(matches.items()))
    else:
        idx = -1
    return idx


def chat_reply(idx, df):
    if idx >= 0:
        print(">>Bot:", df.content[idx])
    else:
        print(">>Bot: I don't have a response for that...")


if __name__ == "__main__":
    main()
