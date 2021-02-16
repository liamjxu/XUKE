import requests
import json
import glob
import csv
from fuzzywuzzy import fuzz

def MAG_get_abstracts(affi,name):

    # Generic parameters
    headers = {"Ocp-Apim-Subscription-Key": "b0dbb13065164b6f8bcec38e89df091e"} # Specific to Liam Xu
    querystring = {"mode":"json%0A"}
    payload = "{}"
    count = str(5)

    # Preprocessing for author's name and affliation
    name = name.strip().replace(', Ph.D.','').replace('Dr. ','')
    name = name.lower().replace('.', '').replace('-', ' ').replace('(', '').replace(')', '').replace(', ', ' ').replace(',',' ')
    affi = affi.lower().replace('--',' ').replace('-', ' ')
    
    # request author information with name and affiliation from MAG
    find_authorID_attr = 'AA.AuN,AA.AuId'
    find_authorID_url = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate?&count={}&expr=Composite(AND(AA.AuN='{}',AA.AfN=='{}'))&attributes={}".format(count, name, affi, find_authorID_attr)
    response = requests.request("GET", find_authorID_url, headers=headers, data=payload, params=querystring)
    if 'entities' not in json.loads(response.text):
        print(name+' not Found')
        return
    response_titles_list = json.loads(response.text)['entities']

    # Match author name with returned AA.AuN to determine author ID
    max_ratio = 0
    for author in response_titles_list[0]['AA']:
        curr_ratio = fuzz.token_sort_ratio(name,author['AuN'])
        if(curr_ratio > max_ratio):
            author_mag_id = author['AuId']
            max_ratio = curr_ratio

    # Find all the papers for that author ID
    find_paper_attr = 'AW,DN,IA'
    find_paper_url = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate?&count={}&expr=Composite(AND(AA.AuId={}))&attributes={}".format(count, str(author_mag_id), find_paper_attr)
    response = requests.request("GET", find_paper_url, headers=headers, data=payload, params=querystring)
    response_titles_list = json.loads(response.text)['entities']

    #Get paper information
    abstract_list = []
    for entity in response_titles_list:
        index_length = entity['IA']['IndexLength']
        inverted_index = entity['IA']['InvertedIndex']
        token_list = _get_abstract_from_IA(index_length, inverted_index)
        abstract = u' '.join(token_list)
        abstract_list.append(abstract)
    return abstract_list


def _get_abstract_from_IA(index_length, inverted_index):
    token_list = [None] * index_length
    for token in inverted_index:
        for ind in inverted_index[token]:
            token_list[ind] = token
    return token_list


if __name__ == '__main__':
    MAG_get_abstracts('University of Illinois at Urbana Champaign','Kevin Chenchuan Chang')
