
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


def find_main(graph):
    for node in graph.nodes.values():
        if node['rel'] == 'root':
            return node
    return None
    
def find_node(word, graph):
    for node in graph.nodes.values():
        # Lemmatizers don't work at its best so try to find a match by using
        # lemmatized word with lemmatized graph word, word with graph word, or
        # lemmatized word with graph word
        if (node["lemma"] == word["lemma"] or node["word"] == word["word"]
            or (node["lemma"] is not None and word["word"] in node["lemma"])):
            return node
    return None
    
def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)
    return results

def what_answer( qgraph, sgraph ):
    qmain = find_main(qgraph)
    snode = find_node(qmain, sgraph)
    if snode is not None:
        print( qmain["word"] )
        print( snode )
        print( "Snode: ",snode["word"] )
        print( sgraph )
        if snode["tag"] == "VBN" or snode["tag"] == "VBD":
            for item in snode["deps"]:
                    address = snode["deps"][item][0]
                    if (sgraph.nodes[address]["tag"] == "NNS" or 
                        sgraph.nodes[address]["tag"] == "NNP" or
                        sgraph.nodes[address]["tag"] == "NN"):
                        deps = get_dependents( sgraph.nodes[address],sgraph)
                        deps = sorted(deps+[sgraph.nodes[address]], 
                                      key=operator.itemgetter("address"))
                        return " ".join(dep["word"] for dep in deps)
    return None


def find_answer( qtext,qgraph, sgraph, q_type):
    qmain = find_main(qgraph)
    snode = find_node(qmain, sgraph)
    if snode is not None:
        if q_type == "What":
            print( qmain["word"] )
            print( snode["word"] )
            print( sgraph )
        for node in sgraph.nodes.values():
            if node.get( 'head', None ) is not None:
                if q_type == "Who":
                    if node['rel'] == "nsubj":
                        deps = get_dependents(node, sgraph)
                        deps = sorted(deps+[node], 
                                      key=operator.itemgetter("address"))    
                        return " ".join(dep["word"] for dep in deps)
                elif q_type == "Where":
                    if node['rel'] == "nmod":
                        deps = get_dependents(node, sgraph)
                        deps = sorted(deps+[node], 
                                      key=operator.itemgetter("address"))    
                        return " ".join(dep["word"] for dep in deps)
    if ( q_type == "Who" ):  # who question main word not in sentence
        if "who is the story about?" in qtext.lower():
            dep = []
            # Get all the nouns in the sentence but avoid pronouns, except for I
            # if its a narration
            for node in sgraph.nodes.values():
                if ((node["rel"] == "nsubj" and node["tag"] != "PRP") 
                     or node["word"] == "I"):
                    dep.append( node )
                    for item in node["deps"]:
                        address = node["deps"][item][0]
                        dep.append( sgraph.nodes[address] )
            dep = sorted( dep, key=operator.itemgetter("address") )
            return " ".join(t["word"] for t in dep)
        else: 
            # the main word in the question does not appear in the given answer
            # sentence, try to find the solution by checking if a word from the 
            # sentence is in the question, if that word's dependency revolves 
            # around a nsubj, dobj, or has nmod relation return it and its 
            # dependents
            for node in sgraph.nodes.values():
                key_word = find_node( node, qgraph ) # word in the question
                if key_word is not None:
                    for item in node["deps"]:
                        address = node["deps"][item][0]
                        if (sgraph.nodes[address]["rel"] == "nsubj" or 
                            sgraph.nodes[address]["rel"] == "dobj" or 
                            sgraph.nodes[address]["rel"] == "nmod"):
                            deps = get_dependents( sgraph.nodes[address],sgraph)
                            deps = sorted(deps+[sgraph.nodes[address]], 
                                          key=operator.itemgetter("address"))
                            return " ".join(dep["word"] for dep in deps)
    return None
# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences


def get_bow(tagged_tokens, stopwords):
    bow = set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])
    return bow


def get_lemmatized(tagged_tokens):

    #Method for getting set of lemmatized words from a tagged set
    #Uses the WordnetLemmatizer
    #Transforms the nltk pos tags into wordnet friendly ones
    #Deletes '?' from the end of each set

    lmtzr = WordNetLemmatizer()
    lemmas = []

    for word, pos in tagged_tokens:
        if pos[:1] == 'J':
            lemmas.append(lmtzr.lemmatize(word, wordnet.ADJ))
        elif pos[:1] == 'V':
            lemmas.append(lmtzr.lemmatize(word, wordnet.VERB))
        elif pos[:1] == 'R':
            lemmas.append(lmtzr.lemmatize(word, wordnet.ADV))
        elif pos[:1] == 'N':
            lemmas.append(lmtzr.lemmatize(word, wordnet.NOUN))
        else:
            lemmas.append(lmtzr.lemmatize(word))
    del lemmas[-1]

    return set(lemmas)


def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i + 1:]


# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
def baseline(qbow, qlemm, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    i = 0
    while i < len(sentences):
        # A list of all the word tokens in the sentence
        sbow = get_bow(sentences[i], stopwords)
        slemm = get_lemmatized(sentences[i])
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        intersectbow = qbow & sbow
        intersectlemm = qlemm & slemm
        overlap = len(intersectlemm)

        answers.append((overlap, sentences[i], i))
        i += 1
    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer in form of tuple, the sentence itself and
    # the index of the sentence in the sentences
    best_answer = ( answers[0][1], answers[0][2] )
    return best_answer

def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str

    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id
    """
    ###     Your Code Goes Here         ###
    if question['type'] == 'Sch':
        stext = story['sch']
    else:
        stext = story["text"]
    qtext = question["text"]
    qlemm = get_lemmatized(get_sentences(qtext)[0])


    stopwords = set(nltk.corpus.stopwords.words("english"))
    moreStopWords = set([",", "."])
    stopwords = stopwords.union(moreStopWords)

    qbow = get_bow(get_sentences(qtext)[0], stopwords)

    sentences = get_sentences(stext)
    answer = baseline(qbow, qlemm, sentences, stopwords)

    answer_graph = ""
    if question['type'] == 'Sch':
        answer_graph = story['sch_dep'][answer[1]]
    else:
        answer_graph = story["story_dep"][answer[1]]

    q_type = qtext.split(" ")[0]
    if q_type == "What":
        print( question["qid"] )
        print( qtext )
    # Answer before this call returns a sentence that most likely has an answer
    # to it. Now try to find the answer inside of the sentence (For precision)
    sub_answer = None
    if q_type != "What":
        sub_answer = find_answer( qtext, question["dep"], answer_graph, q_type )
    else:
        sub_answer = what_answer( question["dep"], answer_graph )
    if sub_answer is not None:
        if q_type == "What":
            answer = " ".join(t[0] for t in answer[0])
            print( answer )
            print( sub_answer )
            print( "--------------------------------------------" )
        return sub_answer
    # No better solution, return the entire sentence
    answer = " ".join(t[0] for t in answer[0])
    if q_type == "What":
        print( answer )
        print( "--------------------------------------------" )
    ###     End of Your Code         ###
    return answer



#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa(evaluate=False):
    QA = QAEngine(evaluate=evaluate)
    QA.run()
    QA.save_answers()

#############################################################


def main():
    # set evaluate to True/False depending on whether or
    # not you want to run your system on the evaluation
    # data. Evaluation data predictions will be saved
    # to hw6-eval-responses.tsv in the working directory.
    run_qa(evaluate=False)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
