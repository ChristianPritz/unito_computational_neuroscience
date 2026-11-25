#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:13:01 2026

@author: wormulon
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt

terms = {
    "Incorrect code": 65,
    "False scientific results": 78,
    "Hallucinations": 70,
    "Over-reliance on AI": 68,
    "Loss of critical thinking": 62,
    "Reproducibility": 62,
    "Data privacy": 23,
    "Research integrity": 52,
    "Loss of coding skills": 53,
    "Unverified outputs": 35,
    "Hidden bugs": 44,
    "Lack of transparency": 35,
    "Invented references": 22,
    "Methodological errors": 45,
    "Scientific accountability": 38,
    "Bias": 39,
    "human obsolence":100
}

wc = WordCloud(
    width=1600,
    height=1000,
    background_color="white",
    colormap="viridis",
    prefer_horizontal=0.85,
    collocations=False,
    margin=12,
    random_state=17,
).generate_from_frequencies(terms)

fig = plt.figure(figsize=(16, 10), dpi=120)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
path = "/home/wormulon/models/unito_computational_neuroscience/connectomics_i/data/scientists_ai_concerns_wordcloud.png"
plt.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
plt.show()