from PIL import Image
import matplotlib.pyplot as plt
import wordcloud as wc
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def generate_word_cloud(_keyword_score_dict, ratio=1):

    # Get a sub-dictionary
    if len(_keyword_score_dict) == 0:
        return np.array([[]]), []
    if ratio != 1:
        key_list = list(_keyword_score_dict.keys())
        key_list.sort(key=lambda x:_keyword_score_dict[x], reverse=True)
        selected_len = int(len(key_list)*ratio)
        key_list = key_list[:selected_len]
        keyword_score_dict = {k:_keyword_score_dict[k] for k in key_list if k in _keyword_score_dict}
    else: 
        key_list = list(_keyword_score_dict.keys())
        key_list.sort(key=lambda x:_keyword_score_dict[x], reverse=True)
        keyword_score_dict = _keyword_score_dict

    # Generate word cloud
    cloud_mask = np.array(Image.open("./resources/blackpic.jpg"))
    wordcloud = wc.WordCloud(width=900,height=500, max_words=1628,relative_scaling=1.0,normalize_plurals=False,mask=cloud_mask).generate_from_frequencies(keyword_score_dict)
    wc_array = wordcloud.to_array()

    return wordcloud, wc_array, key_list

def get_semantic_capacity(key_list):
    with open('./resources/wiki_term_level.json') as json_file:
        semcap_dict = json.load(json_file)
        term_list = list(semcap_dict.keys())
    with open('./resources/wiki_term_level_unk.json') as json_file:
        semcap_dict_unk = json.load(json_file)
        term_list_unk = list(semcap_dict_unk.keys())
    with open('../XUKE/resources/wiki_term_emb_unk.json') as json_file:
        term_emb_unk = np.array(json.load(json_file))
        term_norm_unk = np.linalg.norm(term_emb_unk, axis=1).reshape(-1,1)

    model_name = 'distilbert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    out = {}
    for kw in key_list:
        if kw in semcap_dict:
            out[kw] = semcap_dict[kw]
        else:
            kw_emb = model.encode([kw])
            kw_norm = np.linalg.norm(kw_emb)
            scores = term_emb_unk @ kw_emb.T / term_norm_unk
            idx = np.where(scores==np.amax(scores))[0][0]
            if scores[idx]/kw_norm < 0.6:
                continue
            else:
                out[kw] = semcap_dict_unk[term_list_unk[idx]]
    return out 