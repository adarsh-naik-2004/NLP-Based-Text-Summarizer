import spacy
import json
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from spacy.matcher import Matcher
from spacy import displacy
from IPython.display import Image, display
from spacy import displacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
import re
from spacy.symbols import nsubj, VERB, ADJ
import pandas as pd
import numpy as np
import subprocess
from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize


subprocess.run(["python", "-m", "spacy", "download", "en"])
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)   
# Function to clean text
def clean_text(text):
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t', '', str(text))
    # removing new line characters
    text = re.sub('\n ', '', str(text))
    text = re.sub('\n', ' ', str(text))
    # removing apostrophes
    text = re.sub("'s", '', str(text))
    # removing hyphens
    text = re.sub("-", ' ', str(text))
    text = re.sub("— ", '', str(text))
    # removing quotation marks
    text = re.sub('\"', '', str(text))
    # removing salutations
    text = re.sub("Mr\.", 'Mr', str(text))
    text = re.sub("Mrs\.", 'Mrs', str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    text = text.replace("\r", "")
    return text

def summarize(long_rev):
    # Handle missing values
    long_rev = long_rev.replace('nan', '')
    summ = spacy.load('en_core_web_sm')

    long_rev = summ(long_rev)

    keyword = []
    stopwords = set(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']

    for token in long_rev:
        if token.text.lower() in stopwords or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            keyword.append(token.text)

    freq_word = Counter(keyword)
    max_freq = freq_word.most_common(1)[0][1] if keyword else 1

    # Normalizing the frequency, handle division by zero
    for word in freq_word.keys():
        freq_word[word] = freq_word[word] / max_freq if max_freq != 0 else 0

    sent_strength = {}
    for sent in long_rev.sents:
        # Filter out sentences with fewer than 5 words (adjust as needed)
        if len(sent) >= 5:
            for word in sent:
                if word.text in freq_word.keys():
                    sent_strength[sent] = sent_strength.get(sent, 0) + freq_word[word.text]

    # Select top sentences based on strength
    summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)

    # Return the final summarized review
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)

    return summary

# Function for Information Extraction Operations
def IE_Operations(review):
    doc = nlp(review)
    adjectives = set()
    verbs_all = set()
    pos_tag = ['ADJ', 'VERB']
    print("POS Tagging:")
    for token in doc:
        if token.pos_ not in ["SPACE", "DET", "ADP", "PUNCT", "AUX", "SCONJ", "CCONJ", "PART"]:
            print(f"{token.text} -> {token.pos_}")
        if token.pos_ == "ADJ":
            adjectives.add(token.text)
        if token.pos_ == "VERB":
            verbs_all.add(token.text)
    return adjectives, verbs_all

def calculate_word_frequency(text):
    # Tokenize the text into words
    words = text.split()

    # Remove common words like 'the', 'is', 'are', etc.
    common_words = set(['the', 'is', 'are', 'and', 'it', 'in', 'to', 'of', 'for', 'on', 'with', 'this', 'that','a','i','my','you'])
    filtered_words = [word.lower() for word in words if word.lower() not in common_words]

    # Calculate the frequency of important words
    word_freq = Counter(filtered_words)

    return word_freq


def analyze_sentiment(reviews):
    sentences = [sentence for review in reviews for sentence in sent_tokenize(review)]
    sid = SentimentIntensityAnalyzer()

    positive_keywords = ['good', 'best', 'nice', 'exceptional', 'sensational', 'great', 'excellent', 'perfect', 'wonderful', 'outstanding', 'fantastic']
    negative_keywords = ['worse', 'unacceptable', 'inferior', 'ordinary', 'unsatisfactory','lacks','junk','broke','unreliable','slow','annoying','wasted','overpriced','replace']

    top_positive_sentences = get_top_sentences(sentences, positive_keywords)
    top_negative_sentences = get_top_sentences(sentences, negative_keywords)

    return top_positive_sentences, top_negative_sentences

def get_top_sentences(sentences, keywords, top_n=5):
    matching_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return matching_sentences[:top_n]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file:
            df = pd.read_csv(file)
            if 'Review' not in df.columns:
                return "Error: 'Review' column not found in the CSV file"

            combined_reviews = ' '.join(df['Review'].astype(str))
            adjectives, verbs = IE_Operations(combined_reviews)
            top_positive_sentences, top_negative_sentences = analyze_sentiment(combined_reviews)
            word_freq = calculate_word_frequency(combined_reviews)
            summary = summarize(combined_reviews)
            
            return render_template('result.html', adjectives=adjectives, verbs=verbs, summary=summary,word_freq=word_freq,top_positive_sentences=top_positive_sentences, top_negative_sentences=top_negative_sentences)

    return render_template('index.html')





if __name__ == '__main__':
    app.run(debug=True)
