
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk, operator
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import baseline
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

'''
# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences


def get_bow(tagged_tokens, stopwords):
    bow = set([t[0].lower() for t in tagged_tokens if t[0].lower() not in stopwords])
    return bow

'''
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

def question_restatement(tagged_tokens):
    return None


def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i + 1:]


# qtokens: is a list of pos tagged question tokens with SW removed
# sentences: is a list of pos tagged story sentences
# stopwords is a set of stopwords
'''
def baseline(qbow, qlemm, sentences, stopwords):
    # Collect all the candidate answers
    answers = []
    for sent in sentences:
        # A list of all the word tokens in the sentence
        sbow = get_bow(sent, stopwords)
        slemm = get_lemmatized(sent)
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        intersectbow = qbow & sbow
        intersectlemm = qlemm & slemm
        overlap = len(intersectlemm)

        answers.append((overlap, sent))

    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = answers[0][1]
    return best_answer
'''

def get_answer(question, story):
    answer = baseline.get_the_right_sentence_maybe(question["qid"])
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


    '''
    if question['type'] == 'sch':
        stext = story['sch']
    else:
        stext = story["text"]
    qtext = question["text"]
    qlemm = get_lemmatized(get_sentences(qtext)[0])
    qbow = get_bow(get_sentences(qtext)[0], STOPWORDS)

    #print("question: ", qtext)
    #print("question bow: ", qbow)
    #print("lemmatized version: ", qlemm)

    sentences = get_sentences(stext)
    answer = baseline(qbow, qlemm, sentences, STOPWORDS)
    #print("answer:", " ".join(t[0] for t in answer))
    #print("")
    answer = " ".join(t[0] for t in answer)

    '''

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
