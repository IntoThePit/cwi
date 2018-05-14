# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:19:27 2018

@author: pmfin
"""

from collections import defaultdict

simple_words = defaultdict(int)
simple_bi = defaultdict(int)
simple_tri = defaultdict(int)

with open("1000simpleEnglishWords.txt") as file:
    for line in file:
        for w in line.split(', '):
            simple_words[w] += 1
            for l in range(1, len(w)):
                bi = w[l-1] + w[l]
                simple_bi[bi] += 1
                if l > 1:
                    tri = w[l-2] + w[l-1] + w[l]
                    print(w, tri)
                    simple_tri[tri] += 1
            