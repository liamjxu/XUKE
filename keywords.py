# The main logic of keyword extraction
import nltk
import networkx as nx
import numpy as np
import json
from nltk.stem import WordNetLemmatizer
from itertools import combinations
from queue import Queue
from collections import Counter
from MAG import MAG_get_abstracts
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

###### GLOBAL VARS #####
INCLUDE_POS = ['NN','NNS','JJ']       # Syntax Filters
CONVERGENCE_THRESHOLD = 0.0001  # Convergence threshold
CONVERGENCE_EPOCH_NUM = 50      # force stop after how many epochs
WINDOW_SIZE = 10    # window size used in graph building
DAMPING_FACTOR = 0.85   # damping factor used in graph building
KEYWORD_RATIO = 0.6     # how much keywords do we want to keep for each publication
SCALING = 1.0    # how strong a effect the year has on publication importance, the larger the stronger.
KEYWORD_LIST = json.load(open("cs_keyword_list.json", 'r')) # the keyword list for proper CS words
DIVERSITY = 0.9 # how diverse the keywords are

def keywords(text, ratio=KEYWORD_RATIO, diversity=DIVERSITY):

    # Sanity check
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a string")
    
    # get the tokenized text, all lowercase
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokenized_text = tokenizer.tokenize(text.lower())
    tokenized_text_with_punc = nltk.word_tokenize(text.lower())

    # filter out all the nouns and adjectives in filtered_text
    tagged_text = nltk.pos_tag(tokenized_text)
    filtered_unit = list(filter(lambda x : x[1] in INCLUDE_POS, tagged_text))
    filtered_text = [ x[0] for x in filtered_unit]
    token_pos_dict = {}
    token_lemma_dict = {}
    for x in filtered_unit:
        token_pos_dict[x[0]] = 'a' if x[1] == 'JJ' else 'n' 

    # build the graph
    text_graph = nx.Graph()
    for word in filtered_text:
        if word not in list(text_graph.nodes):
            text_graph.add_node(word)
    
    # set the graph edges with weights (1/2) first window
    lemmatizer = WordNetLemmatizer() # lemmatizer
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
    damping = DAMPING_FACTOR
    if (len(list(text_graph.nodes))==0):
        raise ValueError("Text is empty!")
    lemma_score_dict = dict.fromkeys(text_graph.nodes(), 1/len(list(text_graph.nodes)))
    for _ in range(CONVERGENCE_EPOCH_NUM):
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
            break

    candidates = list(lemma_score_dict.keys())
    candidates.sort(key=lambda w: lemma_score_dict[w], reverse=True)
    selected = candidates[:int(len(candidates)*ratio)]

    # combine keywords
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
    keyword_score_dict = {}
    for comb in combined_list:
        # filter out single adjectives
        if len(comb) == 1 and token_pos_dict[comb[0]] == 'a':
            continue
        comb_lem = [lemmatizer.lemmatize(w.lower()) for w in comb]
        if any(item not in KEYWORD_LIST for item in comb_lem):
            continue
        keyword = ' '.join(comb)
        score = 0
        for word in comb: 
            score += lemma_score_dict[token_lemma_dict[word]]
        keyword_score_dict[keyword] = score

    scattered_keywords = scatter_keywords(list(keyword_score_dict.keys()), text, diversity=diversity)
    final_keyword_score_dict = {k:keyword_score_dict[k] for k in scattered_keywords}

    return final_keyword_score_dict

def keywords_multiple(text_list, ratio=KEYWORD_RATIO, diversity=DIVERSITY):
    if not isinstance(text_list, list):
        raise ValueError('Input is not a list')
    years = [i for (_,i,_) in text_list]
    earliest = min(years)
    latest = max(years)
    citations = [i for (_,_,i) in text_list]
    least = min(citations)
    most = max(citations)

    final_cnt = Counter({})
    for idx, (text, year, citation) in enumerate(text_list):
        print('[keywords_multiple] Processing text ({} of {}).'.format(idx, len(text_list)))
        print('year: {}, citation:{}'.format(year, citation))
        year_score = 1 - (year-earliest)/(latest-earliest)
        citation_score = 1 - (citation-least)/(most-least)
        paper_scaling_factor = np.e**(-1 * year_score * citation_score * SCALING) # how paper_scaling_factor is calculated
        
        keyword_score_dict = keywords(text, ratio, diversity)
        
        keyword_score_dict_scaled = {k:v * paper_scaling_factor for k,v in keyword_score_dict.items()}
        current_cnt = Counter(keyword_score_dict_scaled)
        final_cnt = final_cnt + current_cnt

    keyword_score_dict = dict(final_cnt)
    scattered_keywords = scatter_keywords(list(keyword_score_dict.keys()), ' . '.join([x[0] for x in text_list]), diversity=diversity)
    final_keyword_score_dict = {k:keyword_score_dict[k] for k in scattered_keywords}
    
    print(final_keyword_score_dict)
    return final_keyword_score_dict

def filter_abstracts(abstract_list, time_ratio):
    years = [i for (_,i,_) in abstract_list]
    earliest = min(years)
    latest = max(years)
    starting_year = int(earliest + time_ratio[0] * (latest-earliest))
    ending_year = int(earliest + time_ratio[1] * (latest-earliest))
    filtered_abstracts = [ab for ab in abstract_list if ab[1] >= starting_year and ab[1] <= ending_year]
    return filtered_abstracts, starting_year, ending_year


def _mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int = 5,
        diversity: float = 0.8) -> List[Tuple[str, float]]:

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [(words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]

def scatter_keywords(keywords, document, diversity=0.9, top_ratio=0.5):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens') #'distilbert-base-nli-mean-tokens'
    document_embedding = model.encode(document).reshape(1,-1)
    keyword_embeddings = model.encode(keywords)
    top_n = int(len(keywords)*top_ratio)

    scattered_keyword_value_pair = _mmr(document_embedding, keyword_embeddings, keywords, top_n, diversity)
    scattered_keywords = [x[0] for x in scattered_keyword_value_pair]

    print('keywords number:', len(keywords))
    print('top_ratio:',top_ratio)
    print('top_n:',top_n)
    print('scattered keyword number:', len(scattered_keywords))
    return scattered_keywords
