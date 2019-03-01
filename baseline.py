#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''

import sys, nltk, operator
from qa_engine.base import QABase
porterrrr = nltk.PorterStemmer()
driver = QABase()
preStemmed = {}

# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]

    stemmedSents = []
    for sent in sentences:
        stemmedSents.append([word for word in sent])
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    
    return sentences    

def get_bow(tagged_tokens, stopwords):
    s1 = set([porterrrr.stem(t[0]).lower() for t in tagged_tokens if t[0].lower() not in stopwords])
    s2 = set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])
    superset =  s1.union(s2)
    #print(superset)
    return superset

def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i+1:]

# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
def baseline(qbow, sentences, stopwords, type):
    # Collect all the candidate answers
    answers = []
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        answers.append((overlap, sent))

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer =  None
    for a in reversed(answers[0:3]):
        answer = a[1]
        sent = " ".join(t[0] for t in answer)
        
        print("score: ",a[0], " ", sent)
        #try some wizard magic

        if(type == "Where"):
            locations = ['in', 'along', 'on', 'under', 'near', 'at', 'in front of']
            for l in locations:
                if l in sent: best_answer = a[1]

        if(type == "What"):

            #what WAS _______
            #find was is na significant word.

            # I need to search for What __ [(det)? (NN/NNS)]
            # and then return the sentence with the nn/nns.
            pass

        if(type == "Who"):
            #Who was the story about?
            pass

    if(best_answer == None):
        best_answer = (answers[0])[1]
    return best_answer

def get_the_right_sentence_maybe(question_id):
    
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])

    if("Sch" in q["type"]):
        text = story["sch"]
    else: text = story["text"]
    question = q["text"]
    print("question:", question)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    moreStopWords = set([",", "."])
    stopwords = stopwords.union(moreStopWords)

    #determining type from first word for now
    question_type = get_sentences(question)[0][0][0]
    print(question_type)

    qbow = get_bow(get_sentences(question)[0], stopwords)
    sentences = get_sentences(text)

    answer = baseline(qbow, sentences, stopwords, question_type)
    sent = " ".join(t[0] for t in answer)
    #print("answer:", " ".join(t[0] for t in answer))
    return(sent)

if __name__ == '__main__':
    question_id = "fables-01-2"
    print(get_the_right_sentence_maybe(question_id))

