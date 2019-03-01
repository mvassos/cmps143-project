
from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk, operator, re, sys
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

porterrrr = nltk.PorterStemmer()
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

from baseline import get_the_right_sentence_maybe

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

def question_restatement(tagged_tokens):
    return None


def find_phrase(tagged_tokens, qbow):
    for i in range(len(tagged_tokens) - 1, 0, -1):
        word = (tagged_tokens[i])[0]
        if word in qbow:
            return tagged_tokens[i + 1:]


def get_sentence_index(sentences, target):
    i = 0
    for sent in sentences:
        temp = (" ".join(t[0] for t in sent))
        if temp == target:
            return i
        i += 1


    return None

###ADDING CONSTITUENCY TREE FUNCTIONS (FROM STUB) ###

# See if our pattern matches the current root of the tree
def matches(pattern, root):
    # Base cases to exit our recursion
    # If both nodes are null we've matched everything so far
    if root is None and pattern is None:
        return root

    # We've matched everything in the pattern we're supposed to (we can ignore the extra
    # nodes in the main tree for now)
    elif pattern is None:
        return root

    # We still have something in our pattern, but there's nothing to match in the tree
    elif root is None:
        return None

    # A node in a tree can either be a string (if it is a leaf) or node
    plabel = pattern if isinstance(pattern, str) else pattern.label()
    rlabel = root if isinstance(root, str) else root.label()

    # If our pattern label is the * then match no matter what
    if plabel == "*":
        return root
    # Otherwise they labels need to match
    elif plabel == rlabel:
        # If there is a match we need to check that all the children match
        # Minor bug (what happens if the pattern has more children than the tree)
        for pchild, rchild in zip(pattern, root):
            match = matches(pchild, rchild)
            if match is None:
                return None
        return root

    return None


def pattern_matcher(pattern, tree):
    for subtree in tree.subtrees():
        node = matches(pattern, subtree)
        if node is not None:
            return node
    return None

###END OF CONSTITUENCY FUNCTIONS###

###NEW CONSTITUENCY FUNCTIONS: Manny Vassos###

LOC_PP = set(["in", "on", "at", "to"])

def constituency_search(qtype, tree, qtree):

    print("Entered Constituency Search, Type: ", qtype)
    #print("\nTree of Question: ", qtree)
    #print("\nTree of selected Sentece: ", tree)


    qtype = qtype.lower()

    if qtype == 'where':
        #print("*WHERE*\n")
        # Create our pattern
        pattern = nltk.ParentedTree.fromstring("(VP (*) (PP))")

        # # Match our pattern to the tree
        #print("\nPattern one found: ")
        subtree = pattern_matcher(pattern, tree)

        if subtree is None:
            return None

        #print(" ".join(subtree.leaves()))
        #print("\nSubtree1: ", subtree)

        # create a new pattern to match a smaller subset of subtree
        pattern = nltk.ParentedTree.fromstring("(PP)")
        #print("Pattern two found: ")
        # Find and print the answer
        subtree2 = None
        if subtree is not None:
            subtree2 = pattern_matcher(pattern, subtree)
        if subtree2 is not None:
            ans = (" ".join(subtree2.leaves()))
            #print("ans before: ", ans)
            #for pp in LOC_PP:
             #   if pp in ans:
              #      ans = ans.replace(pp, "")
            #print("ans after: ", ans)
            return ans

    elif qtype == 'why':
        #print("*WHY*\n")
        #create first 'why' pattern looking for because
        pattern = nltk.ParentedTree.fromstring("(SBAR (*))")

        subtree = pattern_matcher(pattern, tree)

        if subtree is None:
            #answer does not include 'becasue'
            pattern = nltk.ParentedTree.fromstring("(VP (*) (S)) ")
            subtree = pattern_matcher(pattern, tree)

            if subtree is not None:
                #isolate (S)
                pattern = nltk.ParentedTree.fromstring("(S)")
                subtree2 = pattern_matcher(pattern, subtree)
                if subtree2 is not None:
                    return (" ".join(subtree2.leaves()))
                else:
                    return None
        else:
            return (" ".join(subtree.leaves()))

    elif qtype == "who":
        #print(" *WHO*\n")
        pattern = nltk.ParentedTree.fromstring("(NP)")
        subtree = pattern_matcher(pattern, tree)
        if subtree is not None:
            return (" ".join(subtree.leaves()))

    elif qtype == "what":
        #print(" *WHAT*\n")
        pattern = nltk.ParentedTree.fromstring("(VP (*) (NP))")
        subtree = pattern_matcher(pattern, tree)

        if subtree is None:
            pattern = nltk.ParentedTree.fromstring("(NP)")
            subtree = pattern_matcher(pattern, tree)

        if subtree is not None:
            #print("Pattern one found, tree: ", subtree)
            return (" ".join(subtree.leaves()))

    elif qtype == "when":
        print(" *WHEN* not implemented\n")

    return None


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
        stree = story['sch_par']
    else:
        stext = story["text"]
        stree = story['story_par']

    qtext = question["text"]
    qtree = question['par']

    question_type = get_sentences(qtext)[0][0][0]

    #print("question: ", qtext)
    #print("question bow: ", qbow)
    #print("lemmatized version: ", qlemm)

    sentences = get_sentences(stext)

    print("*Getting best sentence for ", question['qid'])

    best_sent = get_the_right_sentence_maybe(question['qid'])

    print("Before Constituency Search: ", best_sent, "\n")

    sent_index = get_sentence_index(sentences, best_sent)
    cons_answer = None
    if sent_index is not None:
        cons_answer = constituency_search(question_type, stree[sent_index], qtree)

    print("Consistency Search Results: ", cons_answer)
    if cons_answer is None:
        answer = best_sent
        if answer is None:
            answer = "the best guess"
    else:
        answer = cons_answer

    print("Final Answer: ", answer, "\n")

    ###debugging tool to step through questions!

    #print("Press Any Key To Advance...")

    #try:
     #   print("Exit with q")
      #  quit = input()
       # if quit is 'q':
        #    exit()
        #input("Press enter to continue")
    #except SyntaxError:
       # pass

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
    QA = QAEngine(evaluate=False)
    QA.run()
    QA.save_answers()

#############################################################


def main():
    # set evaluate to True/False depending on whether or
    # not you want to run your system on the evaluation
    # data. Evaluation data predictions will be saved
    # to hw6-eval-responses.tsv in the working directory.
    run_qa(evaluate=True)
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()

if __name__ == "__main__":
    main()
