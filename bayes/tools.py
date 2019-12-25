#!/usr/bin/python3
import string

def line2wordsVect(line):
    words = []
    line = line.strip()
    line = line.strip(string.digits)
    line = line.strip(string.punctuation)
    
    for word in line.split():
        words.append(word)
    return words
