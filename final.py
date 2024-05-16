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
subprocess.run(["python", "-m", "spacy", "download", "en"])
nlp = spacy.load('en_core_web_sm')


def summarize(long_rev):
    summ = spacy.load('en_core_web_sm')
    long_rev = summ(long_rev)
    #print(f"Number of sentences : {len(list(long_rev.sents))}\n")

    keyword = []
    stopwords = list(STOP_WORDS)
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in long_rev:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keyword.append(token.text)
    freq_word = Counter(keyword)
    print("freq",freq_word)
    #print("Filtering tokens \n")
    #print(freq_word.most_common(5))

    # Normalization
    # Each sentence is weighed based on the
    # frequency of the token present in each sentence

    #highest frequency
    max_freq = Counter(keyword).most_common(1)[0][1]
    
    #normalizing the frequency
    for word in freq_word.keys():
        freq_word[word] = (freq_word[word]/max_freq)
    freq_word.most_common(5)

    # Strength of sentences
    sent_strength = {}
    for sent in long_rev.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]
    #print("sentences with their respective strengths \n")
    #print(sent_strength)

# the nlargest function returns a list containing the top 3 sentences which are stored as summarized_sentences

    summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
    #print("top 3 sentences with max strength ")
    #print(summarized_sentences, "\n")

    #print("Final Summarized Review ")
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)
    return summary











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


# text = "The Honda City is a well balanced car with an amazing engine to drive. It runs very smoothly and rarely breaks down"

# knowledge_graph(text)
def IE_Operations(review):
    # create spacy doc
    doc = nlp(review)
    adjectives = set()
    verbs_all = set()
    # applying POS to each token
    print("POS Tagging : ")
    for token in doc:
        if token.pos_ not in ["SPACE", "DET", "ADP", "PUNCT", "AUX", "SCONJ", "CCONJ", "PART"]:
            print(token.text, '->', token.pos_)
        if(token.pos_ == "ADJ"):
            adjectives.add(token.text)
        if(token.pos_ == "VERB"):
            verbs_all.add(token.text)

    print("Dependency Graph : \n")

    print("************************************************************\n")
    displacy.render(doc, style='dep', jupyter=True)
    print("************************************************************\n")

    print("Verb with subject : \n")

    # Finding a verb with a subject
    verbs = set()
    for possible_subject in doc:
        if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
            verbs.add(possible_subject.head)
    print(verbs)
    print("************************************************************\n")

    print("Adjectives : \n")

    # Finding adjectives with a subject
    print(adjectives)
    print("************************************************************\n")
    # NER
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print("************************************************************\n")

    print("Knowledge Graph : \n")

    knowledge_graph(review)

    print("************************************************************\n")
# # function to preprocess speech

# def clean(text):

#     # removing paragraph numbers
#     text = re.sub('[0-9]+.\t', '', str(text))
#     # removing new line characters
#     text = re.sub('\n ', '', str(text))
#     text = re.sub('\n', ' ', str(text))
#     # removing apostrophes
#     text = re.sub("'s", '', str(text))
#     # removing hyphens
#     text = re.sub("-", ' ', str(text))
#     text = re.sub("— ", '', str(text))
#     # removing quotation marks
#     text = re.sub('\"', '', str(text))
#     # removing salutations
#     text = re.sub("Mr\.", 'Mr', str(text))
#     text = re.sub("Mrs\.", 'Mrs', str(text))
#     # removing any reference to outside text
#     text = re.sub("[\(\[].*?[\)\]]", "", str(text))
#     text = text.replace("\r", "")

#     return text
# df = pd.read_csv(r'C:\Users\adars\OneDrive\Desktop\HTML, CSS, JS\minor_iv_sem\datasets\Scraped_Car_Review_ferrari.csv',
#                  delimiter=',', nrows=100)
# df.head()
# # preprocessing speeches
# df['Review_clean'] = df['Review'].apply(clean)
# df.head()
# df['Review_clean'][2]
# reviews = df['Review_clean'][0:10]
# reviews = np.array(reviews)
# reviews[8]
# # applying POS tagging to each review in the dataset and preparing dependency graph, knowledge graph
# # and filtering out verbs and pronouns to apply sentiment analysis as well
# IE_Operations(reviews[0])
# from statistics import mean


# def IE_brand(brand):
#     path = r"C:\Users\adars\OneDrive\Desktop\HTML, CSS, JS\minor_iv_sem\datasets\Scraped_Car_Review_" + brand + ".csv"
#     df = pd.read_csv(path, delimiter=',', nrows=100).sort_values(by = 'Rating', ascending = False)
#     df['Review_clean'] = df['Review'].apply(clean)
#     #df['Review_clean'][2]
#     reviews = df['Review_clean'][0:5]
#     reviews = list(reviews)
#     print("TOP 5 REVIEWS FOR THE BRAND {} ARE:".format(brand.upper()))
#     for i in range(0, 5):
#         print("{}. {}".format(i+1, reviews[i]))
#     review = ' '.join(reviews)
#     mean = df["Rating"].mean()
#     sum = summarize(review)
#     print("\nSUMMARY REVIEW FOR THE {}: \n". format(brand.upper()))
#     print(sum, '\n')
#     IE_Operations(sum)
#     print("MEAN SENTIMENT ASSOCIATED WITH THE BRAND {} : {}!".format(brand.upper(), mean))
# IE_brand("ferrari")