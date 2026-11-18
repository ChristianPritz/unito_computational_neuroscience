#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:45:14 2026

@author: wormulon
"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt

terms = {
    "Incorrect code": 70,
    "False scientific results": 90,
    "Hallucinations": 85,
    "Over-reliance on AI": 83,
    "Loss of critical thinking": 75,
    "Reproducibility": 73,
    "Data privacy": 69,
    "Research integrity": 67,
    "Loss of coding skills": 63,
    "Unverified outputs": 60,
    "Hidden bugs": 50,
    "Lack of transparency": 52,
    "Invented references": 45,
    "Methodological errors": 52,
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
path = "/mnt/data/scientists_ai_concerns_wordcloud.png"
plt.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
plt.show()