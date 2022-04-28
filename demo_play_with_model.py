# coding: utf-8

from __future__ import division, print_function

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import models
import data

import theano
import sys
import re

from io import open

import theano.tensor as T
import numpy as np

from flask import Flask, request

app = Flask(__name__)



numbers = re.compile(r'\d')
is_number = lambda x: len(numbers.sub('', x)) / len(x) < 0.6

model_file = 'Demo-Europarl-EN.pcl'
show_unk = False
x = T.imatrix('x')

print("Loading model parameters...")
net, _ = models.load(model_file, 1, x)

print("Building model...")
predict = theano.function(inputs=[x], outputs=net.y)
word_vocabulary = net.x_vocabulary
punctuation_vocabulary = net.y_vocabulary
reverse_word_vocabulary = {v: k for k, v in net.x_vocabulary.items()}
reverse_punctuation_vocabulary = {v: k for k, v in net.y_vocabulary.items()}

human_readable_punctuation_vocabulary = [p[0] for p in punctuation_vocabulary if p != data.SPACE]
tokenizer = word_tokenize
untokenizer = lambda text: text.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return 'The model is up and running. Send a POST request'
    else:
        return punctuate_api()

@app.route('/punctuate', methods=['POST'])
def punctuate_api():
    # print('begin', file=sys.stderr)
    text = ""
    try:
        try:
            text = request.data.decode('utf-8')
        except NameError:
            text = request.data
        # text = request.json['text']
        print("text: " + text, file=sys.stderr)


        words = [w for w in untokenizer(' '.join(tokenizer(text))).split()
                 if w not in punctuation_vocabulary and w not in human_readable_punctuation_vocabulary]
        # print("words: ", words, file=sys.stderr)
        result = punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, show_unk)
        return result
    except Exception as e:
        print("punctuate_api error: " + str(e), file=sys.stderr)
        return ""
    return "Welcome to machine learning model APIs!"

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return ' '
    elif punct_token.startswith('-'):
        return ' ' + punct_token[0] + ' '
    else:
        return punct_token[0] + ' '

def punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, show_unk):
    result = ""

    if len(words) == 0:
        sys.exit("Input text from stdin missing.")

    if words[-1] != data.END:
        words += [data.END]

    i = 0

    while True:

        subsequence = words[i:i+data.MAX_SEQUENCE_LEN]

        if len(subsequence) == 0:
            return ""
            break

        converted_subsequence = [word_vocabulary.get(
                "<NUM>" if is_number(w) else w.lower(),
                word_vocabulary[data.UNK])
            for w in subsequence]

        if show_unk:
            subsequence = [reverse_word_vocabulary[w] for w in converted_subsequence]

        y = predict(to_array(converted_subsequence))

        # f_out.write(subsequence[0].title())
        # print('result1', file=sys.stderr)
        result += subsequence[0].title()
        # print('result2', file=sys.stderr)
        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(y_t.flatten())
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        if subsequence[-1] == data.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        for j in range(step):
            current_punctuation = punctuations[j]
            # f_out.write(convert_punctuation_to_readable(current_punctuation))
            result += convert_punctuation_to_readable(current_punctuation)
            if j < step - 1:
                if current_punctuation in data.EOS_TOKENS:
                    # f_out.write(subsequence[1+j].title())
                    # print("result3: ", file=sys.stderr)
                    result += subsequence[1+j].title()
                    # print("result4: ", file=sys.stderr)
                else:
                    # f_out.write(subsequence[1+j])
                    # print("result5: ", file=sys.stderr)
                    result += subsequence[1+j]
                    # print("result6: ", file=sys.stderr)

        if subsequence[-1] == data.END:
            return result
            break

        i += step


if __name__ == "__main__":
    app.run(port=5556, host='0.0.0.0')

    # with open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False) as f_out:
    #     while True:
    #         try:
    #             text = raw_input("\nTEXT: ").decode('utf-8')
    #         except NameError:
    #             text = input("\nTEXT: ")
    #
    #         words = [w for w in untokenizer(' '.join(tokenizer(text))).split()
    #                  if w not in punctuation_vocabulary and w not in human_readable_punctuation_vocabulary]
    #
    #         punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, f_out, show_unk)
    #         f_out.flush()
