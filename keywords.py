import nltk
import networkx as nx
from nltk.stem import WordNetLemmatizer
from itertools import combinations
from queue import Queue
import numpy as np
from collections import Counter

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
TEXT_EXAMPLE = """Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."""

lemmatizer = WordNetLemmatizer() # lemmatizer


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
    if (len(list(text_graph.nodes))==0):
        raise ValueError("Text is empty!")
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
        keyword_score_dict[keyword] = score**2


    result_keywords.sort(key=lambda x: keyword_score_dict[x], reverse=True)
    
    wc_array = generate_word_cloud(keyword_score_dict)

    print('------------------Results------------------')
    for i in result_keywords:
        print('{0} ({1})'.format(i, keyword_score_dict[i]))

    return keyword_score_dict, result_keywords, wc_array

def generate_word_cloud(keyword_score_dict):
    # Generate Word Cloud
    print('Generating word cloud...')
    cloud_mask = np.array(Image.open("blackpic.jpg"))
    wordcloud = wc.WordCloud(width=900,height=500, max_words=1628,relative_scaling=0.5,normalize_plurals=False,mask=cloud_mask).generate_from_frequencies(keyword_score_dict)
    wc_array = wordcloud.to_array()
    return wc_array
    # plt.imshow(wordcloud, interpolation='bilinear')

def keywords_multiple(text_list):
    if not isinstance(text_list, list):
        raise ValueError('Input is not a list')

    final_cnt = Counter({})
    for text in text_list:
        keyword_score_dict, _, _ = keywords(text)
        current_cnt = Counter(keyword_score_dict)
        final_cnt = final_cnt + current_cnt

    final_keyword_score_dict = dict(final_cnt)
    
    return final_keyword_score_dict

def generate_word_cloud_with_ratio(_keyword_score_dict, ratio):
    # Generate Word Cloud
    if len(_keyword_score_dict) == 0:
        return np.array([[]]), []
    print(_keyword_score_dict)
    key_list = list(_keyword_score_dict.keys())
    key_list.sort(key=lambda x:_keyword_score_dict[x], reverse=True)
    selected_len = int(len(key_list)*ratio)
    key_list = key_list[:selected_len]
    keyword_score_dict = {k:_keyword_score_dict[k] for k in key_list if k in _keyword_score_dict}
    # print(keyword_score_dict)
    print('Generating word cloud...')
    cloud_mask = np.array(Image.open("blackpic.jpg"))
    wordcloud = wc.WordCloud(width=900,height=500, max_words=1628,relative_scaling=0.5,normalize_plurals=False,mask=cloud_mask).generate_from_frequencies(keyword_score_dict)
    wc_array = wordcloud.to_array()
    return wc_array, key_list
    # plt.imshow(wordcloud, interpolation='bilinear')


if __name__ == '__main__':
    # text = TEXT_EXAMPLE
    default_text_1 = "Graph is an important data representation which appears in a wide diversity of real-world scenarios. Effective graph analytics provides users a deeper understanding of what is behind the data, and thus can benefit a lot of useful applications such as node classification, node recommendation, link prediction, etc. However, most graph analytics methods suffer the high computation and space cost. Graph embedding is an effective yet efficient way to solve the graph analytics problem. It converts the graph data into a low dimensional space in which the graph structural information and graph properties are maximumly preserved. In this survey, we conduct a comprehensive review of the literature in graph embedding. We first introduce the formal definition of graph embedding as well as the related concepts. After that, we propose two taxonomies of graph embedding which correspond to what challenges exist in different graph embedding problem settings and how the existing work addresses these challenges in their solutions. Finally, we summarize the applications that graph embedding enables and suggest four promising future research directions in terms of computation efficiency, problem settings, techniques, and application scenarios."
    default_text_2 = "Witnessing the emergence of Twitter, we propose a Twitter-based Event Detection and Analysis System (TEDAS), which helps to (1) detect new events, to (2) analyze the spatial and temporal pattern of an event, and to (3) identify importance of events. In this demonstration, we show the overall system architecture, explain in detail the implementation of the components that crawl, classify, and rank tweets and extract location from tweets, and present some interesting results of our system."
    default_text_3 = "The Web has been rapidly \"deepened\" by the prevalence of databases online. With the potentially unlimited information hidden behind their query interfaces, this \"deep Web\" of searchable databses is clearly an important frontier for data access. This paper surveys this relatively unexplored frontier, measuring characteristics pertinent to both exploring and integrating structured Web sources. On one hand, our \"macro\" study surveys the deep Web at large, in April 2004, adopting the random IP-sampling approach, with one million samples. (How large is the deep Web? How is it covered by current directory services?) On the other hand, our \"micro\" study surveys source-specific characteristics over 441 sources in eight representative domains, in December 2002. (How \"hidden\" are deep-Web sources? How do search engines cover their data? How complex and expressive are query forms?) We report our observations and publish the resulting datasets to the research community. We conclude with several implications (of our own) which, while necessarily subjective, might help shape research directions and solutions."
    default_text_4 = "Users' locations are important to many applications such as targeted advertisement and news recommendation. In this paper, we focus on the problem of profiling users' home locations in the context of social network (Twitter). The problem is nontrivial, because signals, which may help to identify a user's location, are scarce and noisy. We propose a unified discriminative influence model, named as UDI, to solve the problem. To overcome the challenge of scarce signals, UDI integrates signals observed from both social network (friends) and user-centric data (tweets) in a unified probabilistic framework. To overcome the challenge of noisy signals, UDI captures how likely a user connects to a signal with respect to 1) the distance between the user and the signal, and 2) the influence scope of the signal. Based on the model, we develop local and global location prediction methods. The experiments on a large scale data set show that our methods improve the state-of-the-art methods by 13%, and achieve the best performance."

    default_text_list = [default_text_1, default_text_2, default_text_3, default_text_4]

    final_keyword_score_dict = keywords_multiple(default_text_list)
    print(final_keyword_score_dict)



