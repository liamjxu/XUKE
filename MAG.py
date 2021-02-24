# This code is created based on Jiaying Li's code for achieving researcher's paper counts
# which is shared on Slack on Feb 9th, 2021. 

import requests
import json
import glob
import csv
from fuzzywuzzy import fuzz

# GLOBAL VARIABLES
HEADERS = {"Ocp-Apim-Subscription-Key": "b0dbb13065164b6f8bcec38e89df091e"} # Specific to Liam Xu
QUERYSTRING = {"mode":"json%0A"}
PAYLOAD = "{}"
COUNT = str(200)

# Input:    affi - affiliation of the researcher 
#           name - name of the researcher
# Output:   author_mag_id - MAG ID of the researcher
def MAG_get_AuID(affi, name):
    # Preprocessing for author's name and affliation
    name = name.strip().replace(', Ph.D.','').replace('Dr. ','')
    name = name.lower().replace('.', '').replace('-', ' ').replace('(', '').replace(')', '').replace(', ', ' ').replace(',',' ')
    affi = affi.lower().replace('--',' ').replace('-', ' ')
    
    # request author information with name and affiliation from MAG
    find_authorID_attr = 'AA.AuN,AA.AuId'
    find_authorID_url = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate?&count=100&expr=Composite(AND(AA.AuN='{}',AA.AfN=='{}'))&attributes={}".format(name, affi, find_authorID_attr)
    response = requests.request("GET", find_authorID_url, headers=HEADERS, data=PAYLOAD, params=QUERYSTRING)
    if 'entities' not in json.loads(response.text):
        print(name+' not Found')
        return
    entity_list = json.loads(response.text)['entities']
    if len(entity_list) == 0:
        print('entity list is empty')
        return 

    # Match author name with returned AA.AuN to determine author ID
    max_ratio = 0
    for author in entity_list[0]['AA']:
        curr_ratio = fuzz.token_sort_ratio(name,author['AuN'])
        if(curr_ratio > max_ratio):
            author_mag_id = author['AuId']
            max_ratio = curr_ratio

    return author_mag_id


def _get_abstract_from_IA(index_length, inverted_index):
    token_list = [None] * index_length
    for token in inverted_index:
        for ind in inverted_index[token]:
            token_list[ind] = token
    token_list = list(filter(None, token_list)) 
    abstract = u' '.join(token_list)
    return abstract

# Input:    affi - affiliation of the researcher 
#           name - name of the researcher
# Output:   abstract_list - a list of abstract of the researcher's publication
def MAG_get_abstracts(affi,name):
    # get author ID in MAG
    author_mag_id = MAG_get_AuID(affi,name)
    # Find all the papers for that author ID
    find_paper_attr = 'AW,DN,IA'
    find_paper_url = "https://api.labs.cognitive.microsoft.com/academic/v1.0/evaluate?&count={}&expr=Composite(AND(AA.AuId={}))&attributes={}".format(COUNT, str(author_mag_id), find_paper_attr)
    response = requests.request("GET", find_paper_url, headers=HEADERS, data=PAYLOAD, params=QUERYSTRING)
    if 'entities' not in json.loads(response.text):
        return []
    entity_list = json.loads(response.text)['entities']

    #Get paper information
    abstract_list = []
    for entity in entity_list:
        if 'IA' not in entity:
            continue
        index_length = entity['IA']['IndexLength']
        inverted_index = entity['IA']['InvertedIndex']
        abstract = _get_abstract_from_IA(index_length, inverted_index)
        abstract_list.append(abstract)
    return abstract_list


if __name__ == '__main__':
    abstract_list = MAG_get_abstracts('University of Illinois at Urbana Champaign','Kevin Chenchuan Chang')
    for ab in abstract_list:
        print(ab)
        print('\n\n\n')
