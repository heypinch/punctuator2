# coding: utf-8

from __future__ import division

from nltk.tokenize import word_tokenize

import models
import data

import theano
import sys
import codecs
import re

import theano.tensor as T
import numpy as np
import os

numbers = re.compile(r'\d')
is_number = lambda x: len(numbers.sub('', x)) / len(x) < 0.6
env = os.environ

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

    outarr = []
    if len(words) == 0:
        raise RuntimeError("Input text from stdin missing.")

    if words[-1] != data.END:
        words += [data.END]

    i = 0

    while True:

        subsequence = words[i:i+data.MAX_SEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(
                "<NUM>" if is_number(w) else w.lower(),
                word_vocabulary[data.UNK])
            for w in subsequence]

        if show_unk:
            subsequence = [reverse_word_vocabulary[w] for w in converted_subsequence]

        y = predict(to_array(converted_subsequence))

        outarr.append(subsequence[0].title())

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
            outarr.append(convert_punctuation_to_readable(current_punctuation))
            if j < step - 1:
                if current_punctuation in data.EOS_TOKENS:
                    outarr.append(subsequence[1+j].title())
                else:
                    outarr.append(subsequence[1+j])

        if subsequence[-1] == data.END:
            break

        i += step
    return outarr



model_file = env['MODEL_FILE']

show_unk = os.getenv('SHOW_UNK', False)

x = T.imatrix('x')

print "Loading model parameters..."
net, _ = models.load(model_file, 1, x)

print "Building model..."
predict = theano.function(inputs=[x], outputs=net.y)
word_vocabulary = net.x_vocabulary
punctuation_vocabulary = net.y_vocabulary
reverse_word_vocabulary = {v:k for k,v in net.x_vocabulary.items()}
reverse_punctuation_vocabulary = {v:k for k,v in net.y_vocabulary.items()}

human_readable_punctuation_vocabulary = [p[0] for p in punctuation_vocabulary if p != data.SPACE]
tokenizer = word_tokenize
untokenizer = lambda text: text.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")


while True:
    text = raw_input("\nTEXT: ").decode('utf-8')

    words = [w for w in untokenizer(' '.join(tokenizer(text))).split()
             if w not in punctuation_vocabulary and w not in human_readable_punctuation_vocabulary]

    punced = punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, show_unk)
    print(''.join(punced))


# if __name__ == "__main__":
    # app.logger.info("Starting punctuator...")
    # app.run()

