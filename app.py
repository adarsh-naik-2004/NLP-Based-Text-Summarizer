# app.py
from flask import Flask, render_template, request
from final import summarize
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
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import subprocess
from spacy import displacy

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("indexhome.html")

@app.route("/index1.html",methods=["GET", "POST"])
def first_page():
    if request.method == "POST":
        data = request.form["data"]

        # Use the summarize function from final.py
        summary = summarize(data)

        return render_template("index1.html", result=summary)
    else:
        return render_template("index1.html")


@app.route("/index2.html",methods=["GET", "POST"])
def second_page():
    if request.method == 'POST':
        if 'fileInput' not in request.files:
            return "No file part"

        file = request.files['fileInput']
        if file.filename == '':
            return "No selected file"
        
        subprocess.run(["python", "-m", "spacy", "download", "en"])
        nlp = spacy.load('en_core_web_sm')
        summ = spacy.load('en_core_web_sm')
        
        
        def summarize(long_rev):
                        # Handle missing values
            long_rev = long_rev.replace('nan', '')

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
        
        
        def clean(text):

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
            def get_top_sentences(sentences, keywords, top_n=5):
                matching_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
                return matching_sentences[:top_n]
            
            top_positive_sentences = get_top_sentences(sentences, positive_keywords)
            top_negative_sentences = get_top_sentences(sentences, negative_keywords)

            return top_positive_sentences, top_negative_sentences
        
        
        
        
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
        
        
        
        
        
        def getSentences(text):
            nlp = English()
            nlp.add_pipe('sentencizer')
            document = nlp(text)
            return [sent.text.strip() for sent in document.sents]


        def printToken(token):
            print(token.text, "->", token.dep_)


        def appendChunk(original, chunk):
            return original + ' ' + chunk


        def isRelationCandidate(token):
            deps = ["ROOT", "adj", "attr", "agent", "amod"]
            return any(subs in token.dep_ for subs in deps)


        def isConstructionCandidate(token):
            deps = ["compound", "prep", "conj", "mod"]
            return any(subs in token.dep_ for subs in deps)


        def processSubjectObjectPairs(tokens):
            subject = ''
            object = ''
            relation = ''
            subjectConstruction = ''
            objectConstruction = ''
            for token in tokens:
                #printToken(token)
                if "punct" in token.dep_:
                    continue
                if isRelationCandidate(token):
                    relation = appendChunk(relation, token.lemma_)
                if isConstructionCandidate(token):
                    if subjectConstruction:
                        subjectConstruction = appendChunk(
                            subjectConstruction, token.text)
                    if objectConstruction:
                        objectConstruction = appendChunk(
                            objectConstruction, token.text)
                if "subj" in token.dep_:
                    subject = appendChunk(subject, token.text)
                    subject = appendChunk(subjectConstruction, subject)
                    subjectConstruction = ''
                if "obj" in token.dep_:
                    object = appendChunk(object, token.text)
                    object = appendChunk(objectConstruction, object)
                    objectConstruction = ''

            #print(subject.strip(), ",", relation.strip(), ",", object.strip())
            return (subject.strip(), relation.strip(), object.strip())


        def processSentence(sentence):
            tokens = nlp(sentence)
            return processSubjectObjectPairs(tokens)


        def printGraph(triples):
            G = nx.Graph()
            for triple in triples:
                G.add_node(triple[0])
                G.add_node(triple[1])
                G.add_node(triple[2])
                G.add_edge(triple[0], triple[1])
                G.add_edge(triple[1], triple[2])

            pos = nx.spring_layout(G)
            plt.figure()
            nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
                    node_size=500, node_color='seagreen', alpha=0.9,
                    labels={node: node for node in G.nodes()})
            plt.axis('off')
            plt.show()


        def knowledge_graph(text):
            sentences = getSentences(text)
            nlp_model = spacy.load('en_core_web_sm')
            triples = []
            print(text)
            for sentence in sentences:
                triples.append(processSentence(sentence))

            printGraph(triples)
        
        
        def generate_dependency_graph(review):
            # create spacy doc
            doc = nlp(review)
            
            # Get HTML markup for dependency graph
            html_markup = displacy.render(doc, style='dep', page=True)
            
            # Return HTML markup for dependency graph
            return html_markup
        
        
  
        df = pd.read_csv(file,delimiter=',',nrows=100)
        df['Review'] = df['Review'].astype(str)
        df['summary'] = df['Review'].apply(summarize)
        sentiment=df['Review'].astype(str).tolist()
        all_reviews = ' '.join(df['Review'].astype(str).fillna(''))
        df['Review_clean'] = df['Review'].apply(clean)
        all_cleaned_reviews = ' '.join(df['Review_clean'].astype(str))
        df['Review_clean'] = df['Review'].apply(clean)
        
        
        combined_reviews = ' '.join(df['Review'].astype(str))
        df['Review_clean'] = df['Review'].apply(clean)
        all_cleaned_reviews = ' '.join(df['Review_clean'].astype(str))
        word_freq = calculate_word_frequency(combined_reviews)
        summary = summarize(all_cleaned_reviews)
        top_positive_sentences, top_negative_sentences = analyze_sentiment(sentiment)
        word_freq = calculate_word_frequency(all_cleaned_reviews)
        
        adjectives,verbs_all=IE_Operations(all_cleaned_reviews)
        
        knowledge_graph_data = knowledge_graph(summary)
        dependency_graph_data = generate_dependency_graph(summary)
        
              
        
        return render_template('index2.html',summary=summary,top_positive_sentences=top_positive_sentences, top_negative_sentences=top_negative_sentences,word_freq=word_freq,adjectives=adjectives,verbs_all=verbs_all,knowledge_graph_data=knowledge_graph_data,dependency_graph_data=dependency_graph_data)
        
    return render_template("index2.html")

if __name__ == "__main__":
    app.run(debug=True)