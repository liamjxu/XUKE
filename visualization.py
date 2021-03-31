from PIL import Image
import matplotlib.pyplot as plt
import wordcloud as wc
import numpy as np

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
