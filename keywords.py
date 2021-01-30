import nltk
import networkx as nx
from nltk.stem import WordNetLemmatizer
from itertools import combinations
from queue import Queue
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import wordcloud as wc

###### GLOBAL VARS #####
INCLUDE_POS = ['NN','NNS','JJ']       # Syntax Filters
CONVERGENCE_THRESHOLD = 0.0001  # Convergence threshold
CONVERGENCE_EPOCH_NUM = 50      # force stop after how many epochs
WINDOW_SIZE = 10    # window size 
DAMPING_FACTOR = 0.85
KEYWORD_RATIO = 0.6


lemmatizer = WordNetLemmatizer() # lemmatizer

text = """ Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."""
# text = """In this paper, we introduce TextRank – a graph-based ranking model for text processing, and show how this model can be successfully used in natural language applications. In particular, we propose two innova- tive unsupervised methods for keyword and sentence extraction, and show that the results obtained com- pare favorably with previously published results on established benchmarks."""
def keywords(text):
    print('-----------------Initiated-----------------')
    print('Convergence threshold: {}'.format(CONVERGENCE_THRESHOLD))
    print('Maximum Iterating Epoch Nums: {}'.format(CONVERGENCE_EPOCH_NUM))
    print('The window size used to build edges: {}'.format(WINDOW_SIZE))
    print('The damping factor used in Page Rank: {}'.format(DAMPING_FACTOR))
    print('The ratio of keywords kept: {}'.format(KEYWORD_RATIO))
    print('Preprocessing...')
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a string")
    
    # get the tokenized text, all lowercase
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokenized_text = tokenizer.tokenize(text.lower())
    tokenized_text_with_punc = nltk.word_tokenize(text.lower())
    # tokenized_text = nltk.word_tokenize(text)
    # print(tokenized_text)
    # print(tokenized_text_with_punc)



    # filter out all the nouns and adjectives in filtered_text
    tagged_text = nltk.pos_tag(tokenized_text)
    # print('tagged text:')
    # print(tagged_text)
    # print('\n\n')
    filtered_unit = list(filter(lambda x : x[1] in INCLUDE_POS, tagged_text))
    filtered_text = [ x[0] for x in filtered_unit]
    token_pos_dict = {}
    token_lemma_dict = {}
    for x in filtered_unit:
        token_pos_dict[x[0]] = 'a' if x[1] == 'JJ' else 'n' 

    print('Builidng Graph...')
    # build the graph
    text_graph = nx.Graph()
    for word in filtered_text:
        if word not in list(text_graph.nodes):
            text_graph.add_node(word)
    
    # set the graph edges with weights (1/2) first window
    first_window = tokenized_text[:WINDOW_SIZE]
    for word1, word2 in combinations(first_window, 2):
        if word1 in filtered_text and word2 in filtered_text:
            word1_lemma = lemmatizer.lemmatize(word1,pos=token_pos_dict[word1])
            word2_lemma = lemmatizer.lemmatize(word2,pos=token_pos_dict[word2])
            text_graph.add_edge(word1_lemma,word2_lemma,weight=1)
            if word1 not in token_lemma_dict:
                token_lemma_dict[word1] = word1_lemma
            if word2 not in token_lemma_dict:
                token_lemma_dict[word2] = word2_lemma
    
    # set the graph edges with weights (2/2) next windows
    queue = Queue()
    current_window = []
    for i in first_window:
        queue.put(i)
        current_window.append(i)
    for i in range(WINDOW_SIZE, len(tokenized_text)):
        head = queue.get()
        current_window.remove(head)
        tail = tokenized_text[i]
        queue.put(tail)
        current_window.append(tail)
        for word1, word2 in combinations(current_window, 2):
            if word1 in filtered_text and word2 in filtered_text:
                word1_lemma = lemmatizer.lemmatize(word1,pos=token_pos_dict[word1])
                word2_lemma = lemmatizer.lemmatize(word2,pos=token_pos_dict[word2])
                text_graph.add_edge(word1_lemma,word2_lemma,weight=1)
                if word1 not in token_lemma_dict:
                    token_lemma_dict[word1] = word1_lemma
                if word2 not in token_lemma_dict:
                    token_lemma_dict[word2] = word2_lemma

    # conduct the page rank
    print('PageRanking...')
    damping = DAMPING_FACTOR
    lemma_score_dict = dict.fromkeys(text_graph.nodes(), 1/len(list(text_graph.nodes)))
    for epoch in range(CONVERGENCE_EPOCH_NUM):
        convergence_achieved = 0
        for i in text_graph.nodes:
            rank = 1 - damping
            for j in text_graph.adj[i]:
                neighbors_sum = sum(text_graph.edges[j, k]['weight'] for k in text_graph.adj[j])
                rank += damping * lemma_score_dict[j] * text_graph.edges[j,i]['weight'] / neighbors_sum

            if abs(lemma_score_dict[i] - rank) <= CONVERGENCE_THRESHOLD:
                convergence_achieved += 1

            lemma_score_dict[i] = rank

        if convergence_achieved == len(text_graph.nodes()):
            print('stopping epoch num:', epoch)
            break

    candidates = list(lemma_score_dict.keys())
    candidates.sort(key=lambda w: lemma_score_dict[w], reverse=True)
    selected = candidates[:int(len(candidates)*KEYWORD_RATIO)]

    # combine keywords
    print('Generating keywords...')
    combined_list = []
    current_combination = []
    for token in tokenized_text_with_punc: 
        if token not in token_lemma_dict:
            if len(current_combination) != 0:
                if current_combination not in combined_list:
                    combined_list.append(current_combination)
                current_combination = []
            continue
        if token_lemma_dict[token] not in selected:
            if len(current_combination) != 0:
                if current_combination not in combined_list:
                    combined_list.append(current_combination)
                current_combination = []
            continue
        else: 
            current_combination.append(token)

    # output keywords
    print('Combining keywords, generating key phrases...')
    result_keywords = []
    keyword_score_dict = {}
    for comb in combined_list:
        # filter out single adjectives
        if len(comb) == 1 and token_pos_dict[comb[0]] == 'a':
            continue
        keyword = ''
        for idx in range(len(comb)-1):
            keyword += comb[idx]
            keyword += ' '
        keyword += comb[-1]
        result_keywords.append(keyword)
        # calculate keyword score
        score = 0
        for word in comb: 
            score += lemma_score_dict[token_lemma_dict[word]]
        keyword_score_dict[keyword] = score


    result_keywords.sort(key=lambda x: keyword_score_dict[x], reverse=True)
    # return keyword_score_dict, result_keywords

    # Generate Word Cloud
    print('Generating word cloud...')
    cloud_mask = np.array(Image.open("blackpic.jpg"))
    wordcloud = wc.WordCloud(width=900,height=500, max_words=1628,relative_scaling=0.5,normalize_plurals=False,mask=cloud_mask).generate_from_frequencies(keyword_score_dict)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    print('------------------Results------------------')
    for i in result_keywords:
        print('{0} ({1})'.format(i, keyword_score_dict[i]))

keywords(text)



