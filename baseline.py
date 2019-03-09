#!/usr/bin/env python
'''
Created on May 14, 2014
@author: reid

Modified on May 21, 2015
'''

import sys, nltk, operator
from qa_engine.base import QABase
porterrrr = nltk.PorterStemmer()
#porterrrr = nltk.stem.snowball.SnowballStemmer("english")
driver = QABase()
preStemmed = {}

debug = False
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
    s1 = set([toPresentTense(porterrrr.stem(t[0]).lower()) for t in tagged_tokens if t[0].lower() not in stopwords])
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
def baseline(qbow, sentences, stopwords, question):
    q_sents = get_sentences(question)
    type = q_sents[0][0][0]

    # Collect all the candidate answers
    answers = []
    subject = None
    for sent in sentences:
        # A list of all the word tokens in the sentence
        
        #let's try some subject replacement
        #print(sent)
        if(sent[0][1] == "NNP"):
            subject = sent[0]
        newSent = []
        for word in sent:
            if word[0].lower() in ['he', 'she'] and subject is not None:
                if(debug): print("replacing")
                newSent.append(subject)
            else:
                newSent.append(word)
            if word[1] == "NNP":
                #subject = word
                pass
        sent = newSent


        sbow = get_bow(sent, stopwords)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        if debug: 
            print("QBOW: ", qbow)
            print("SBOW: ", sbow)
        overlap = len(qbow & sbow)
        answers.append((overlap, sent))

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer =  None

    bestNounCount = [0,None]

    foundHighPriorityLocation = False

    for a in reversed(answers[0:3]):



        #find the index of the subject based on first occurence
        subj = None
        verb = None
        for word in q_sents[0][1: (len(q_sents[0]) -1)]:
            if(word[1] == "VBD" and verb == None):
                verb =word[0]
            if(word[1] == "NNP" and subj == None):
                subj =word[0]


        answer = a[1]
        sent = " ".join(t[0] for t in answer)
        if debug: print("score: ",a[0], " ", sent)
        #try some wizard magic

        if(type == "Where"):
            locations = ['in', 'upon', 'inside', 'along', 'on', 'to the', 'under', 'near', 'at', 'in front of', 'by the', 'house', 'to']
            superLocations = [" " + l + " " for l in locations] + ["," + l +" " for l in locations]
            for l in superLocations:
                if l in sent:
                    best_answer = a[1]
                    foundHighPriorityLocation = True
                    if debug: print("HPL: ", l)
            for l in locations:
                if l in sent and not foundHighPriorityLocation:
                    best_answer = a[1]
                    if debug: print("LPL: ", l)
        if(type == "What"):
            #print(qbow)
            '''
            if debug: print("QUESTION: ", q_sents[0])
            if debug: print("SUBJECT: ", subj)
            if verb is not None:
                if verb == "did":
                    if debug: print("VERB IS DID!!")
                    subj =  q_sents[0][2][0]
                    verb2 = q_sents[0][3][0]
                    if subj == "the":
                        subj += (" " + q_sents[0][3][0])
                        verb2 = q_sents[0][4][0]

                    phrase = subj + " " + verb2

                    print(phrase)
                    if phrase in sent:
                        print("YEEEEEEEETUSSSSSSSS")
                        best_answer = a[1]

                if(verb in sent):
                    pass
                    #best_answer = a[1]
                if subj is not None:
                    if subj + " verb" in sent:
                        best_answer = a[1]
            '''

            #what WAS _______
            #find was is na significant word.

            # I need to search for What __ [(det)? (NN/NNS)]
            # and then return the sentence with the nn/nns.
            pass
        
        if(type == "Who"):
            if("who is the story about" in question.lower()):
                #return the sentence with the most nouns?
                #print(question)
                #print("score: ",a[0], " ", answer)
                nouncount = 0
                for word in answer:
                    if(word[1] in ["NNP", "NNS"]):
                        nouncount += 1

                if(nouncount > bestNounCount[0]):
                    bestNounCount[0] = nouncount
                    bestNounCount[1] = a[1]

        if(type == "Why"):
            reasons = ["because"]
            for r in reasons:
                if r in sent:
                    best_answer = a[1]
            pass
        if(type == "Did"):
            #sorry mom
            best_answer = [("yes","yes")]

    #for "who was the story about" case
    if bestNounCount[0] != 0:
        best_answer = bestNounCount[1]
        #print("YEE")

    if(best_answer == None):
        best_answer = (answers[0])[1]
    return best_answer

#nltk sucks
def toPresentTense(word):
    translations = {
        "felt": "feel",
        "ate": "eat",
        "mother": "mom",
        "freed": "free",
        "brought": "bring",
        "hid": "hide",
        "fell": "fall",
        "spat": "spit",
        "went": "go",
        "gave": "give",
        "hid":"hide",
        "got": "get",
        "stood": "stand",
        "ran": "run",
        "fed": "feed",
        "had": "have",
        "heard": "hear",
        "stood": "stand",
        "met": "meet",
        "strike": "attack",
        "revolt": "rebel",
        "combust": "burn",
    }
    if word in translations:
        return translations[word]
    else:
        return word



def get_the_right_sentence_maybe(question_id):
    
    q = driver.get_question(question_id)
    story = driver.get_story(q["sid"])

    if("Sch" in q["type"]):
        text = story["sch"]
    else: text = story["text"]
    question = q["text"]
    if debug: print("question:", question)
    stopwords = set(nltk.corpus.stopwords.words("english"))
    moreStopWords = set([",", ".", "?", "'s"])
    stopwords = stopwords.union(moreStopWords)

    qbow = get_bow(get_sentences(question)[0], stopwords)
    sentences = get_sentences(text)

    if debug: print("BASELINE FOR QUESTION: ", question_id)

    answer = baseline(qbow, sentences, stopwords, question)
    sent = " ".join(t[0] for t in answer)
    #print("answer:", " ".join(t[0] for t in answer))
    return sent

if __name__ == '__main__':
    question_id = "fables-01-2"
    print(get_the_right_sentence_maybe(question_id))

