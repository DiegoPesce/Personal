import numpy as np
import os

TRIES = 6
LEN = int(input("\nInserisci la lunghezza della parola: "))
fname = "4mln_no_ax_length%i"%LEN
words = np.load(os.getcwd()+"\\Personal\\PYTHON\\Hangman\\processed_data\\%s.npy"%fname)
probs = { 
    "a": 11.74,
    "b": 0.92,
    "c": 4.50,
    "d": 3.73,
    "e": 11.79,
    "f": 0.95,
    "g": 1.64,
    "h": 1.54,
    "i": 11.28,
    #"j": ,
    #"k": ,
    "l": 6.51,
    "m": 2.51,
    "n": 6.88,
    "o": 9.83,
    "p": 3.05,
    "q": 0.51,
    "r": 6.37,
    "s": 4.98,
    "t": 5.62,
    "u": 3.01,
    "v": 2.10,
    #"x": ,
    #"y": ,
    #"w": ,
    "z": 0.49     
    }	

def reduce(words, letter, positions):
    if positions is None:
        words = [wrd for wrd in words if letter not in wrd]
    else:
        for pos in positions:
            words = [wrd for wrd in words if wrd[pos] == letter]
        words = [wrd for wrd in words if wrd.count(letter) == len(positions)]

    return words

def recalculate_probs(words, probs):
    # Compute the frequency of each letter contained in probs relatively to words.
    n_words = len(words)
    for key in probs:
        probs[key] = 0

    for wrd in words:
        for l in set(wrd):
            if l in probs:
                probs[l] += 1


    to_delete = []
    for key, value in probs.items():
        if value == 0:
            to_delete.append(key)
        else:
            probs[key] = value/n_words

    for key in to_delete:
        del probs[key]

    return probs


# true word = p a n n o c c h i a
# positions = 0 1 2 3 4 5 6 7 8 9
###
# true word = d  i  s  s  i  m  i  l  a  n  d  o  v  i  s  i
# positions = 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
###
# true word = a r t i s t i c o
# positions = 0 1 2 3 4 5 6 7 8

probs = recalculate_probs(words, probs)

while TRIES > 0:
    letter = max(probs, key=probs.get)
    try:
        pos = None
        pos = input("\n Se la lettera \"%s\" è contenuta nella tua parola, dimmi in che posizioni (partendo da 0), altrimenti premi invio: "%letter).split(' ')
        pos = list(map(int,pos))
    except ValueError:
        pos = None
    
    del probs[letter]

    words = reduce(words, letter, pos)

    if pos is None:
        TRIES -= 1

    n_words = len(words)
    if n_words <= TRIES:
        print("Mi rimangono %i tentativi e solo %i parole, la tua parola è una di queste: "%(TRIES,n_words))
        print(words)
        break
    
    probs = recalculate_probs(words, probs)

    print("\nMi rimangono %i parole!"%n_words)



