from sentence_transformers import SentenceTransformer
from nltk import pos_tag, word_tokenize
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
from nltk.stem import WordNetLemmatizer
from MAG import MAG_get_abstracts
import numpy as np
import json

class Profile():
    def __init__(self, name, affi, model='distilbert-base-nli-mean-tokens', diversity=0.7):
        self.name = name
        self.affi = affi
        self.model = SentenceTransformer(model)
        self.keyword_list = json.load(open("./resources/cs_keyword_list.json", 'r')) # the keyword list for proper CS words
        self.ab_idx = 0
        self.diversity = diversity
        keys = json.load(open("./resources/key.json", 'r'))
        values = json.load(open("./resources/freq.json", 'r'))
        self.dictionary = dict(zip(keys,values))
        self.basis_words = json.load(open('./resources/basis_words.json','r'))
        unprocessed_basis = json.load(open("./resources/basis.json", 'r'))
        mean_base = np.mean(unprocessed_basis, axis=0)
        unprocessed_basis-=mean_base
        self.basis = unprocessed_basis/np.linalg.norm(unprocessed_basis).reshape(-1,1)
        self.mean_base = mean_base
        self.keyword_score_dict = {}

        
    
    def get_abstracts_from_MAG(self):
        self.abstracts = MAG_get_abstracts(self.affi, self.name)
    
    def filter_abstracts(self,time_ratio=(0.67,1)):
        years = [i for (_,i,_) in self.abstracts]
        earliest = min(years)
        latest = max(years)
        starting_year = int(earliest + time_ratio[0] * (latest-earliest))
        ending_year = int(earliest + time_ratio[1] * (latest-earliest))
    
        citation = [i for (_,_,i) in self.abstracts]
        most = max(citation)
        least = min(citation)
        
        filtered_abstracts = [ab for ab in self.abstracts if ab[1] >= starting_year and ab[1] <= ending_year]

        self.abstracts = filtered_abstracts
        self.starting_year = starting_year
        self.ending_year = ending_year
        self.earliest = earliest
        self.latest = latest
        self.most = most 
        self.least = least

    def extract_keywords(self, keyword_ratio=0.5):
        self.ab_idx = 0
        NPOS = ['NN', 'NNS']
        lemmatizer = WordNetLemmatizer()
        # for each abstract, extract its keywords
        final_cnt = Counter({})
        for ab, year, citation in self.abstracts:
            # get all candidate_phrases of form JJ*NN+ and in the keyword list 
            tokenized_text_with_punc = word_tokenize(ab.lower())
            tagged_text = pos_tag(tokenized_text_with_punc)
            candidate_phrases = []
            current_phrase = []
            last_tag = '@'
            for word, tag in tagged_text:
                if tag == 'JJ' and (last_tag not in NPOS):
                    current_phrase.append(word)
                    last_tag = tag
                elif tag in NPOS:
                    current_phrase.append(word)
                    last_tag = tag
                else:
                    if len(current_phrase) != 0:
                        comb_lem = [lemmatizer.lemmatize(w.lower()) for w in current_phrase]
                        if all(item in self.keyword_list for item in comb_lem):
                            candidate_phrases.append(' '.join(current_phrase))
                    current_phrase = []
                    last_tag = '@'
            if len(current_phrase) != 0:
                candidate_phrases.append(' '.join(current_phrase))
            if len(candidate_phrases) == 0:
                continue
            # scatter the keywords, assign scores to these keywords by scaling the document
            scattered_keyword_value_pair = self.scatter_keywords(candidate_phrases, ab, diversity=self.diversity, top_ratio=keyword_ratio)
            year_score = 1 - (year-self.earliest)/(self.latest-self.earliest)
            citation_score = 1 - (citation-self.least)/(self.most-self.least)
            paper_scaling_factor = np.e**(-1 * year_score * citation_score)
            current_keyword_score_dict_scaled = defaultdict(lambda:0,{})
            for k, v in scattered_keyword_value_pair:
                k_lem = lemmatizer.lemmatize(k)
                current_keyword_score_dict_scaled[k_lem] += v * paper_scaling_factor 
            current_keyword_score_dict_scaled = dict(current_keyword_score_dict_scaled)
            current_cnt = Counter(current_keyword_score_dict_scaled)
            final_cnt = final_cnt + current_cnt

        unprocessed_keyword_score_dict = dict(final_cnt)

        keyword_score_dict = {}
        for k in unprocessed_keyword_score_dict.keys():
            if self._non_generic(k):
                keyword_score_dict[k] = unprocessed_keyword_score_dict[k]
        self.keyword_score_dict = keyword_score_dict
        return keyword_score_dict
            

    def evaluate_subfields(self):
        k_list = list(self.keyword_score_dict.keys())
        k_list = k_list[:int(len(k_list)*1/3)]
        word_embeddings = self.model.encode(k_list)
        subfield_score = np.sum(self.basis @ word_embeddings.T, axis=0)
        subfield_score /= np.sum(subfield_score)
        return subfield_score




    def _mmr(self,
            doc_embedding: np.ndarray,
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

    def _non_generic(self,word):
        word_ = self.model.encode(word)-self.mean_base
        c = self.basis @ word_.T 
        c = np.abs(c)
        max_val = np.max(c)
        min_val = np.min(c)
        if max_val/min_val > 500:
            return True
        else:
            return False


    def scatter_keywords(self, keywords, document, diversity, top_ratio):
        document_embedding = self.model.encode(document).reshape(1,-1)
        keyword_embeddings = self.model.encode(keywords)
        # top_n = int(len(keywords)*top_ratio)
        top_n = len(keywords)

        scattered_keyword_value_pair = self._mmr(document_embedding, keyword_embeddings, keywords, top_n, diversity)
        scattered_keywords = [x[0] for x in scattered_keyword_value_pair]

        print('processing paper ({} of {})'.format(self.ab_idx, len(self.abstracts)))
        self.ab_idx+=1
        print('final keyword number:', len(scattered_keywords))
        return scattered_keyword_value_pair


