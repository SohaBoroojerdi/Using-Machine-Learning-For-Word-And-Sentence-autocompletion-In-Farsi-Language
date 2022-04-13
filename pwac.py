# Persian Word AND sentence Auto Completion using machine learning algorithms #


from operator import add
import string
from turtle import bgcolor
import numpy as np
import pandas as pd
import re
import PySimpleGUI as sg
import tensorflow as tf
from tensorflow import keras
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
import time



title = "Persian Word and Sentence Auto Completion"
sg.theme('DarkBlue 3')

### Font parameters
titlefont = ("Britannic", 15)
titlefontcolor = "black"
fontsize = 20

####################################################################################################
## DATASET PARAMETERS
####################################################################################################

### corpus
corpusname = "cp.txt"
corpusnames = ["Bijankhan_Sample.txt", "Hamshahri_Sample.txt", "Bijankhan.txt", "Hamshahri.txt", "Varzeshi.txt", "LSTM_train.txt"]
corpuslimit = 100000

### dataset
dataset = []
datasetname = "ds.txt"
datasetnames = ["Blank Dataset", "Corpus ("+corpusname+")", "Dataset ("+datasetname+")"]

### Word size
### Minimum number of letters per word
minwordlen = 2 

### Maximum number of letters per word 
maxwordlen = 5  

### Word completion algorithm
algorithm = "Tree"

### KNN parameters
### Maximum number of selected condidate words (k)
knn_topk = 5  
knn_mode = 1

### LSTM parameters
### Sentence completion parameters
LSTM_sequencelen = 4
LSTM_numcandidates = 5
LSTM_numoutputs = 128
LSTM_validation_split = 0.05
LSTM_batch_size = 128
LSTM_epochs = 2
LSTM_modelname = "LSTM_model.h5"
LSTM_historyname = "LSTM_history.p"

LSTM_modelsource = ["Rebuild from '"+corpusname+"'", "Load '"+LSTM_modelname+"'"]

LSTM_unique_words = []
LSTM_unique_word_indices = {}

LSTM_prev_words = []
LSTM_next_words = []

### Display parameterts
msglist = []

### Number of messages displaying in the GUI
msglength = 4



####################################################################################################
## USER INTERFACE
####################################################################################################

### Window layout
### Left column
l_col = [
    [
        sg.Drop(
        key='algorithm',
        enable_events=True,
        readonly=True, size=(21),
        values=('Tree', 'KNN'),
        default_value=algorithm,
        font=fontsize)
    ],

    [
        sg.Text(
        key='knn_text',
        text="Nearest Neighbors (k):",
        text_color="gray" if (algorithm!="KNN") else "white",
        font=fontsize),

        #sg.InputText(
        # key='knn_topk',
        # enable_events=True,
        # default_text=str(knn_topk),
        # disabled=(algorithm!='KNN'),
        # size=(3),
        # font=fontsize),

        sg.Spin(
            key='knn_topk',
            enable_events=True,
            readonly=True,
            values=[x for x in range(1, 10)],
            initial_value=str(knn_topk),
            disabled=(algorithm!='KNN'),
            size=(3),
            font=fontsize),
    ],

    [
        sg.Radio(
            key='knn_mode1',
            enable_events=True,
            group_id="KNN",
            text="dist = depth",
            default=(knn_mode==1),
            disabled=(algorithm!='KNN'),
            font=fontsize)
    ],

    [
        sg.Radio(
            key='knn_mode2',
            enable_events=True,
            group_id="KNN",
            text="dist = 1/frequency",
            default=(knn_mode==2),
            disabled=(algorithm!='KNN'),
            font=fontsize)
    ],

    [
        sg.Radio(
            key='knn_mode3',
            enable_events=True,
            group_id="KNN",
            text="dist = depth/frequency",
            default=(knn_mode==3),
            disabled=(algorithm!='KNN'),
            font=fontsize)],
]

### Right column
r_col = [
    [
        sg.InputText(
            key='keyword',
            enable_events=True,
            disabled=True,
            size=(15),
            justification="right",
            font=fontsize),

        sg.Button(
            key='addword',
            button_text='Add Word',
            disabled=True,
            font=fontsize),
    ],

    [
        sg.Listbox(
            key='wordlist',
            enable_events=True,
            size=(25, 5),
            values=[],
            font=fontsize)],
]

### Main layout
layout = [
    [
        sg.Text(
            text="..:: Welcome to "+title+" ::..\nSoha Boroojerdi -- Utah Valley University -- 2022",
            size=(60, 2),
            justification="center",
            text_color="black",
            font=fontsize)
    ],

    #[
    # sg.Text(
    # key='message',
    # text_color="red",
    # size=(60,2),
    # justification="center",
    # text='',
    # font=fontsize)],
    [
        sg.Text(text="Messages",
        text_color=titlefontcolor,
        font=titlefont),
    ],

    [
        sg.Listbox(
            key='messages',
            no_scrollbar=True,
            size=(60, msglength),
            values=msglist,
            background_color="lightgray",
            font=fontsize)
    ],

    [
        sg.Text(text="Corpus Creation",
        text_color=titlefontcolor,
        font=titlefont),
    ],

    [
        sg.Text(
            text="Select up to",
            font=fontsize),

        sg.InputText(
            key='corpuslimit',
            default_text=corpuslimit,
            size=(8),
            font=fontsize),

        sg.Text(
            text="words from",
            font=fontsize),

        sg.Drop(
            key='corpus',
            readonly=True,
            size=(15),
            values=(corpusnames),
            default_value=corpusnames[0],
            font=fontsize),

        sg.Button(
            key='extract',
            button_text='Extract',
            font=fontsize),
    ],

### Word completion section
    [
        sg.Text(
            text="Word Completion",
            text_color=titlefontcolor,
            font=titlefont),
    ],

    [
        sg.Text(
            text="Words with",
            font=fontsize),

        #sg.InputText(
        # key='minwordlen',
        # default_text=str(minwordlen),
        # size=(3),
        # font=fontsize),

        sg.Spin(
            key='minwordlen',
            readonly=True,
            values=[x for x in range(1, 10)],
            initial_value=str(minwordlen),
            size=(1),
            font=fontsize),

        sg.Text(
            text="to",
            font=fontsize),

        #sg.InputText(
        # key='maxwordlen',
        # default_text=str(maxwordlen),
        # size=(3),
        # font=fontsize),

        sg.Spin(
            key='maxwordlen',
            readonly=True,
            values=[x for x in range(1, 10)],
            initial_value=str(maxwordlen),
            size=(1),
            font=fontsize),

        sg.Text(
            text="letters",
            font=fontsize),

        sg.Drop(
            key='dataset',
            readonly=True,
            size=(10),
            values=(datasetnames),
            default_value=datasetnames[0],
            font=fontsize),

        sg.Button(
            key='load',
            button_text='Load',
            font=fontsize),

        #sg.Button(
        # key='blank',
        # button_text='Blank',
        # font=fontsize),

        sg.Button(
            key='save',
            button_text='Save',
            disabled=True,
            font=fontsize),
    ],
 
    #[sg.Text(
    # text="Word :",
    # text_color="black",
    # font=fontsize)],
    [
        sg.Column(l_col),
        sg.VSeperator(),
        sg.Column(r_col)
    ],

    ### Sentence completion section
    [
        sg.Text(
            text="Sentence Completion",
            text_color=titlefontcolor,
            font= titlefont),
    ],

    [
        sg.Text(
            text="Source of the LSTM model",
            font=fontsize),

        sg.Drop(
            key='lstm_model',
            readonly=True,
            size=(24),
            values=(LSTM_modelsource),
            default_value=LSTM_modelsource[0],
            font=fontsize),

        sg.Button(
            key='lstm_load',
            button_text='Load',
            font=fontsize),
    ],

    [
        sg.Text(
            text="Sequence Length",
            font=fontsize),

        sg.Spin(
            key='LSTM_sequencelen',
            readonly=True,
            values=[x for x in range(3, 10)],
            initial_value=str(LSTM_sequencelen),
            size=(3),
            font=fontsize),

        sg.Text(
            text="   Candidates",
            font=fontsize),

        sg.Spin(
            key='lstm_numcandidates',
            readonly=True,
            values=[x for x in range(1, 10)],
            initial_value=str(LSTM_numcandidates),
            size=(3),
            font=fontsize),
    ],

    [
        sg.Text(
            text="Validation Split     ",
            font=fontsize),

        sg.InputText(
            key='LSTM_validation_split',
            default_text=LSTM_validation_split,
            justification='left',
            size=(4),
            font=fontsize),

        sg.Text(
            text="    Neurons     ",
            font=fontsize),

        sg.InputText(
            key='LSTM_numoutputs',
            default_text=LSTM_numoutputs,
            justification='right',
            size=(4),
            font=fontsize),
    ],

    [
        sg.Text(
            text="Batch Size            ",
            justification='left',
            font=fontsize),

        sg.InputText(
            key='LSTM_batch_size',
            default_text=LSTM_batch_size,
            justification='left',
            size=(4),
            font=fontsize),

        sg.Text(
            text="    Epochs       ",
            justification='right',
            font=fontsize),

        sg.InputText(
            key='LSTM_epochs',
            default_text=LSTM_epochs,
            justification='right',
            size=(4),
            font=fontsize),
    ],

    [
        sg.InputText(
        key='lstm_sentence',
        enable_events=True,
        disabled=True,
        size=(60),
        justification="right",
        font=fontsize)
    ],

    [
        sg.Listbox(
            key='lstm_wordlist',
            enable_events=True,
            size=(60, 8),
            values=[],
            font=fontsize)
    ],

    #[
    # sg.Button(
    # 'Exit',
    # key='exit')],
]


### Create the window
window = sg.Window(title, layout, size=(502, 700)).Finalize()


####################################################################################################
## DISPLAY FUNCTIONS FOR TERMINAL/GUI
####################################################################################################

### Display a Message
def displayMessage(message):
    print(">> "+message)
    msglist.append(">> "+message)
    while len(msglist) > msglength:
        msglist.pop(0)
    window['messages'].update(msglist)
    window.refresh()


### Display Words of a List for Tree
def displayWords(words):
    wordlist = []
    for word in words:
        print(word[0])
        wordlist.append(word[0])
    window['wordlist'].update(wordlist)


### Display Words of a List for LSTM
def displayLSTMWords(words):
    for word in words:
        print(word)
    window['lstm_wordlist'].update(words)


####################################################################################################
## WORD TREE DATA STRUCTURE
####################################################################################################

### Tree Node
class Tree:
    def __init__(self):
        self.endofword = False
        self.count = 0
        self.letter = ""
        self.children = []


### List Word Candidates from a Node of the Tree
def listWords(node, str, depth=0):
    words = []
    if node.endofword:
        words.append([str, depth, node.count])
    for child in node.children:
        new_str = str + child.letter
        words = words + listWords(child, new_str, depth+1)
    return words


### Add a Node to the Tree
def addWord(root, word):
    #print(word)
    for i in range(len(word)):
        #print("\t", i, "/", len(word), "\t", word[i])
        found = False
        for j in range(len(root.children)):
            #print("\t\t", j, "/", len(root.children))
            letter = root.children[j].letter
            if word[i] == letter:
                root = root.children[j]
                found = True
                break
        if not found:
            temp = Tree()
            temp.letter = word[i]
            root.children.append(temp)
            root = root.children[-1]
            #print("traverse: "); traverse(root)
    root.count = root.count + 1
    root.endofword = True


### Find a Node of the Tree
def findNode(node, word):
    found = True
    #print(word)
    for i in range(len(word)):
        #print("\t", i, "/", len(word), "\t", word[i])
        match = False
        for j in range(len(node.children)):
            #print("\t\t", j, "/", len(node.children), node.children[j].letter)
            if word[i] == node.children[j].letter:
                node = node.children[j]
                match = True
                break
        if not match:
            found = False
            break
    return found, node


wordForest = Tree()


####################################################################################################
## WORD LIST OPERATIONS
####################################################################################################

### Rank Words for KNN
def rankWords(words, mode):
    rankedWords = []
    for word in words:
        ### the following is for defining distance; word[0]: word, word[1]: depth, and word[2]: count
        if mode == 1:       # based on depth
            rankedWords.append([word[0], word[1]])
        elif mode == 2:     # based on 1/inverse
            rankedWords.append([word[0], 1/word[2]])
        elif mode == 3:     # based on depth/count
            rankedWords.append([word[0], word[1]/word[2]])

    ###  Sorting the rankedWordws 
    for i in range(len(rankedWords)-1):
        for j in range(i+1, len(rankedWords)):
            if rankedWords[i][1] > rankedWords[j][1]:
                temp = rankedWords[i]
                rankedWords[i] = rankedWords[j]
                rankedWords[j] = temp
    return rankedWords


### Find Top Words for KNN
def topWords(words, k):
    selectedWords = []
    for word in words:
        if k > 0:
            selectedWords.append(word)
            k = k - 1
        else:
            break
    return selectedWords


####################################################################################################
## SENTENCE PROCESSING FUNCTIONS
####################################################################################################

### Converting text to vector
def text2vector(text):
    base = 0
    found = True
    vector = np.zeros((1, LSTM_sequencelen, len(LSTM_unique_words)))
    if len(text.split()) < LSTM_sequencelen:
        base = LSTM_sequencelen - len(text.split())
    #print(base)
    for t, word in enumerate(text.split()):
        if word in LSTM_unique_word_indices.keys():
            vector[0, base+t, LSTM_unique_word_indices[word]] = 1
            #print(t,word)
        #else:
            #found = False
            #break
    return found, vector

### Find top LSTM words
def topLSTM(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

### LSTM prediction
def predictLSTM(text, n=3):
    if text == "":
        return []
    found, vector = text2vector(text)
    if found:
        preds = LSTM_model.predict(vector, verbose=0)[0]
        next_indices = topLSTM(preds, n)
        return [LSTM_unique_words[idx] for idx in next_indices]
    else:
        return []


####################################################################################################
## OLD MAIN BODY
####################################################################################################

### Merging multiple txt file(s)
### Set to False to skip this step
if False:
    filename1 = "Bijankhan_Sample.txt"
    filename2 = "Hamshahri_Sample.txt"
    displayMessage("Merging multiple corpus files ...")
    data = []
    with open(filename1, encoding="utf-8") as infile1:
        data = data + infile1.read().split()
    with open(filename2, encoding="utf-8") as infile2:
        data = data + infile2.read().split()
    #data_farsi = [word for word in data if re.findall("[آ-ی]|[گ]|[چ]|[پ]|[ژ]", word)]
    data_farsi = []
    for word in data:
        allfarsi = True
        for i in range(len(word)):
            if not re.findall("[آ-ی]|[گ]|[چ]|[پ]|[ژ]", word[i]):
                allfarsi = False
                break
        if allfarsi:
            data_farsi.append(word)
    #print(data_farsi)
    #data_distinct = list(dict.fromkeys(data_farsi))


    ### Creating new dataset file
    with open(datasetname, mode="w", encoding="utf-8") as outfile:
        for word in data_farsi:
            outfile.write("%s\n" % word)


###  load dataset from file
if False:
    displayMessage("Loading dataset from file '"+datasetname+"' ...")
    with open(datasetname, encoding="utf-8") as outfile:
        dataset = outfile.read().split()


###  create word tree
if False:
    displayMessage("Building a forest of word trees ...")
    wordForest = Tree()
    for word in dataset:
        if(len(word) >= minwordlen) and (len(word) <= maxwordlen):
            addWord(wordForest, word)

while False:
    key = input("Enter your 'partial' word : ")
    if key == "quit" or key == "q":
        break


    ###  search for a key
    found, node = findNode(wordForest, key)

    if found:
        ###  find candidates
        print("== Candidates from Tree ...")
        words = listWords(node, key)
        displayWords(words)


        ###  find k-nearest candidates
        print("== Candidates from KNN ...")
        print(":: distance = depth")
        selectedwords = topWords(rankWords(words, 0), knn_topk)
        displayWords(selectedwords)
        print(":: distance = 1/count")
        selectedwords = topWords(rankWords(words, 1), knn_topk)
        displayWords(selectedwords)
        print(":: distance = depth/count")
        selectedwords = topWords(rankWords(words, 2), knn_topk)
        displayWords(selectedwords)

    else:
        print("No candidate word found.")


####################################################################################################
## MAIN BODY
####################################################################################################

# Create an event loop
while True:
    event, values = window.read()
    ### Print(values)
    print(":: event :: ",event," ::")

    refresh = False
    lstm_refresh = False

    ### Extract persian words
    if event == "extract":
        filename = values['corpus']
        displayMessage("Reading file '" + filename + "' ...")
        data = []
        with open(filename, encoding="utf-8") as infile:
            data = data + infile.read().replace(".", " ").replace("؟", " ").replace("!", " ").split()
        data_farsi = []
        displayMessage("Extracting persian words from '" + filename + "' ...")
        for word in data:
            allfarsi = True
            for i in range(len(word)):
                ### Only keeps Farsi words
                if not re.findall("[آ-ی]|[گ]|[چ]|[پ]|[ژ]", word[i]):
                    allfarsi = False
                    break
            if allfarsi:
                data_farsi.append(word)


        ### Creating a new file for persian corpus
        displayMessage("Creating persian corpus '" + corpusname + "' ...")
        corpuslimit = int(values['corpuslimit'])
        corpuscount = 0
        with open(corpusname, mode="w", encoding="utf-8") as outfile:
            for word in data_farsi:
                if corpuscount >= corpuslimit:
                    break
                outfile.write("%s\n" % word)
                corpuscount += 1
        displayMessage("Corpus '" + corpusname + "' with "+ str(corpuscount) + " words was created.")


    ### Add word to dataset
    elif event == "addword":
        word = values['keyword']
        if(len(word) >= minwordlen) and (len(word) <= maxwordlen):
            addWord(wordForest, word)
            displayMessage("word '" + word + "' was added to the dataset.")
            refresh = True
        else:
            displayMessage("Word length is out of bounds ("+str(minwordlen)+", "+str(maxwordlen)+").")


    ### Load dataset
    elif event == "load":
        minwordlen = int(values['minwordlen'])
        maxwordlen = int(values['maxwordlen'])
        if minwordlen < maxwordlen:
            ds = values['dataset']

            ### Make a blank dataset
            if ds == datasetnames[0]:       
                dataset = []
            
            ### Load from corpus
            elif ds == datasetnames[1]:     
                displayMessage("Loading dataset from '" + corpusname + "' ...")
                with open(corpusname, encoding="utf-8") as outfile:
                    dataset = outfile.read().split()

            ### Load from dataset
            elif ds == datasetnames[2]:     
                displayMessage("Loading dataset from '" + datasetname + "' ...")
                with open(datasetname, encoding="utf-8") as outfile:
                    dataset = outfile.read().split()

            ###  Create word tree
            displayMessage("Building a forest of word trees ...")
            wordForest = Tree()
            numWords = 0
            for word in dataset:
                if(len(word) >= minwordlen) and (len(word) <= maxwordlen):
                    addWord(wordForest, word)
                    numWords+=1
            displayMessage("Forest of word trees with "+str(numWords)+" words was built.")
            window['keyword'].update(disabled=False)
            window['save'].update(disabled=False)

        else:
            displayMessage("Invalid word length bounds ("+str(minwordlen)+", "+str(maxwordlen)+") defined.")

    ### Save dataset
    elif event == "save":
        ###  Save dataset to ds.txt file
        displayMessage("Saving dataset to file ('" + datasetname + "') ...")
        with open(datasetname, mode="w", encoding="utf-8") as outfile:
            words = listWords(wordForest, "")
            for word in words:
                for i in range(word[2]):
                    outfile.write("%s\n" % word[0])
        displayMessage("Dataset was saved to '" + datasetname + "'.")

    ### Selected algorithm received
    elif event == "algorithm":
        if algorithm != values['algorithm']:
            algorithm = values['algorithm']
            window['wordlist'].update([])

            ### Enable/disable KNN algorithm
            window['knn_topk'].update(disabled=(algorithm!="KNN"))
            window['knn_text'].update(text_color="gray" if (algorithm!="KNN") else "white")
            window['knn_mode1'].update(disabled=(algorithm!="KNN"))
            window['knn_mode2'].update(disabled=(algorithm!="KNN"))
            window['knn_mode3'].update(disabled=(algorithm!="KNN"))
            refresh = True

    ### KNN top k value received
    elif event == "knn_topk":
        knn_topk = int(values['knn_topk'])
        refresh = True

    ### KNN mode 1 value received 
    ### mode 1 => dist=depth
    elif event == "knn_mode1":
        knn_mode = 1
        refresh = True

    ### KNN mode 2 value received 
    ### mode 2 => dist=1/frequency
    elif event == "knn_mode2":
        knn_mode = 2
        refresh = True

    ### KNN mode 3 value received 
    ### mode 3 => dist=depth/frequency
    elif event == "knn_mode3":
        knn_mode = 3
        refresh = True

    ### Selected word from wordlist received 
    elif event == "wordlist":
        if len(values['wordlist']) > 0:
            select = values['wordlist'][0]
            if select != "":
                values['keyword'] = select
                window['keyword'].update(select)
                window['wordlist'].update([])
                refresh = True

    ### Exit signal received
    elif event == "exit" or event == sg.WIN_CLOSED:
        break

    ### Input from the text box received
    if event == "keyword" or refresh:
        key = values['keyword']
        window['wordlist'].update([])
        if key != "":
            ###  Search for a key
            startTree = time.time()
            print("Require time for finding a node: \n")
            found, node = findNode(wordForest, key)
            
            if found:
                ###  Find candidates
                words = listWords(node, key)
                endTree = time.time()
                print(endTree - startTree)
                if algorithm == "Tree":
                    print("== Showing word candidates for Tree ...")
                    displayWords(words)

                ###  Find k-nearest candidates
                elif algorithm == "KNN":
                    startKNN = time.time()
                    print("Require time for find a node and a KNN search: \n")
                    selectedwords = topWords(rankWords(words, knn_mode), knn_topk)
                    endKNN = time.time()
                    totalKNN_time = (endKNN - startKNN) + (endTree - startTree)
                    print(totalKNN_time)

                    print("== Showing word candidates for KNN ...")
                    displayWords(selectedwords)

                else:
                    displayMessage(algorithm + " not yet implemented!")

            else:
                displayMessage("No match found!")

            window['addword'].update(disabled=False)

        else:
            window['addword'].update(disabled=True)


####################################################################################################
## LSTM
####################################################################################################

    # Load 
    if event == "lstm_load":
        displayMessage("Loading words from '"+corpusname+"' for LSTM ...")
        ### Using tokenizers to split each word and store them
        tokenizer = RegexpTokenizer(r'\w+')
        words = open(corpusname).read()
        #print("Input file: \n", words)
        LSTM_words = tokenizer.tokenize(words)
        print("After Tokenize: \n", LSTM_words)
        

        displayMessage("Creating a list of unique words for LSTM ...")

        ### Getting unique words
        LSTM_unique_words = np.unique(tokenizer.tokenize(open(corpusname).read()))
        LSTM_unique_word_indices = dict((c, i) for i, c in enumerate(LSTM_unique_words))
        print("LSTM Dictionary: \n", LSTM_unique_word_indices)
        displayMessage(str(len(LSTM_unique_words))+" unique words were loaded for LSTM.")

        ms = values['lstm_model']
        if ms == LSTM_modelsource[0] :
            displayMessage("Preparing dataset for a new model build ...")

            LSTM_sequencelen = int(values['LSTM_sequencelen'])
            LSTM_numoutputs = int(values['LSTM_numoutputs'])
            LSTM_validation_split = float(values['LSTM_validation_split'])
            LSTM_batch_size = int(values['LSTM_batch_size'])
            LSTM_epochs = int(values['LSTM_epochs'])
           
            ### Feature Engineering to make words into numerical represntations
            for i in range(len(LSTM_words) - LSTM_sequencelen):
                LSTM_prev_words.append(LSTM_words[i:i + LSTM_sequencelen])
                LSTM_next_words.append(LSTM_words[i + LSTM_sequencelen])
            print("Previous words at 8: \n", LSTM_prev_words[8])
            print("Next words at 8: \n", LSTM_next_words[8])

            ### Storing features(X) and labels(Y)
            ### iterate X (previous) and Y (next) if the word is available so that the corresponding position becomes 1
            X = np.zeros((len(LSTM_prev_words), LSTM_sequencelen, len(LSTM_unique_words)), dtype=bool)
            Y = np.zeros((len(LSTM_next_words), len(LSTM_unique_words)), dtype=bool)
            for i, each_words in enumerate(LSTM_prev_words):
                for j, each_word in enumerate(each_words):
                    X[i, j, LSTM_unique_word_indices[each_word]] = 1
                Y[i, LSTM_unique_word_indices[LSTM_next_words[i]]] = 1
            print("X: \n", X)
            print("Y:\n", Y)

            ### Building a LSTM model
            displayMessage("Building a new LSTM model ...")
            LSTM_model = Sequential()
            LSTM_model.add(LSTM(LSTM_numoutputs, input_shape=(LSTM_sequencelen, len(LSTM_unique_words))))
            LSTM_model.add(Dense(len(LSTM_unique_words)))
            LSTM_model.add(Activation('softmax'))

            optimizer = RMSprop(lr=0.01)
            LSTM_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            LSTM_model.summary()
            #history = LSTM_model.fit(X, Y, validation_split=LSTM_validation_split, batch_size=LSTM_batch_size, epochs=LSTM_epochs, shuffle=True).history

            ### Training the model
            LSTM_model.fit(X, Y, validation_split=LSTM_validation_split, batch_size=LSTM_batch_size, epochs=LSTM_epochs, shuffle=True)
            print("Size X: ", len(X), "     Size Y: ", len(Y))

            LSTM_model.save(LSTM_modelname)
            #pickle.dump(history, open(LSTM_historyname, "wb"))
            displayMessage("The LSTM model '"+LSTM_modelname+"' was built.")

        ### Loading LSTM_model.h5 file
        elif ms == LSTM_modelsource[1] :
            displayMessage("Loading LSTM model '"+LSTM_modelname+"' ...")
            LSTM_model = load_model(LSTM_modelname)
            #history = pickle.load(open(LSTM_historyname, "rb"))
            displayMessage("The LSTM model '"+LSTM_modelname+"' was loaded.")

        window['lstm_sentence'].update(disabled=False)

    ### Selected word from LSTM wordlist received 
    elif event == "lstm_wordlist":
        if len(values['lstm_wordlist']) > 0:
            select = values['lstm_wordlist'][0]
            if select != "":
                values['lstm_sentence'] = values['lstm_sentence'] + (select + " ")
                window['lstm_sentence'].update(values['lstm_sentence'])
                #window['wordlist'].update([])
                lstm_refresh = True

    if event == "lstm_sentence" or lstm_refresh:
        window['lstm_wordlist'].update([])
        text = values['lstm_sentence']
        if text != "" and text[-1] == " ":
            LSTM_sequencelen = int(values['LSTM_sequencelen'])
            LSTM_numcandidates = int(values['lstm_numcandidates'])
            texttokens = tokenizer.tokenize(text)
            
            ### Take the last tokens of texttokens for prediction (based on the sequence lenght value) 
            if len(texttokens) > LSTM_sequencelen:
                sequence = " ".join(texttokens[-LSTM_sequencelen:])
            ### Take all the tokens 
            else:
                sequence = text

            ### LSTM prediction
            #displayMessage("Sequence: " + sequence)
            displayLSTMWords(predictLSTM(sequence, LSTM_numcandidates))

window.close()