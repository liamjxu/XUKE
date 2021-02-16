import streamlit as st
from keywords import keywords, keywords_multiple, generate_word_cloud_with_ratio

st.markdown("# Keyword Extraction based on TextRank:")

default_text_1 = "Graph is an important data representation which appears in a wide diversity of real-world scenarios. Effective graph analytics provides users a deeper understanding of what is behind the data, and thus can benefit a lot of useful applications such as node classification, node recommendation, link prediction, etc. However, most graph analytics methods suffer the high computation and space cost. Graph embedding is an effective yet efficient way to solve the graph analytics problem. It converts the graph data into a low dimensional space in which the graph structural information and graph properties are maximumly preserved. In this survey, we conduct a comprehensive review of the literature in graph embedding. We first introduce the formal definition of graph embedding as well as the related concepts. After that, we propose two taxonomies of graph embedding which correspond to what challenges exist in different graph embedding problem settings and how the existing work addresses these challenges in their solutions. Finally, we summarize the applications that graph embedding enables and suggest four promising future research directions in terms of computation efficiency, problem settings, techniques, and application scenarios."
default_text_2 = "Witnessing the emergence of Twitter, we propose a Twitter-based Event Detection and Analysis System (TEDAS), which helps to (1) detect new events, to (2) analyze the spatial and temporal pattern of an event, and to (3) identify importance of events. In this demonstration, we show the overall system architecture, explain in detail the implementation of the components that crawl, classify, and rank tweets and extract location from tweets, and present some interesting results of our system."
default_text_3 = "The Web has been rapidly \"deepened\" by the prevalence of databases online. With the potentially unlimited information hidden behind their query interfaces, this \"deep Web\" of searchable databses is clearly an important frontier for data access. This paper surveys this relatively unexplored frontier, measuring characteristics pertinent to both exploring and integrating structured Web sources. On one hand, our \"macro\" study surveys the deep Web at large, in April 2004, adopting the random IP-sampling approach, with one million samples. (How large is the deep Web? How is it covered by current directory services?) On the other hand, our \"micro\" study surveys source-specific characteristics over 441 sources in eight representative domains, in December 2002. (How \"hidden\" are deep-Web sources? How do search engines cover their data? How complex and expressive are query forms?) We report our observations and publish the resulting datasets to the research community. We conclude with several implications (of our own) which, while necessarily subjective, might help shape research directions and solutions."
default_text_4 = "Users' locations are important to many applications such as targeted advertisement and news recommendation. In this paper, we focus on the problem of profiling users' home locations in the context of social network (Twitter). The problem is nontrivial, because signals, which may help to identify a user's location, are scarce and noisy. We propose a unified discriminative influence model, named as UDI, to solve the problem. To overcome the challenge of scarce signals, UDI integrates signals observed from both social network (friends) and user-centric data (tweets) in a unified probabilistic framework. To overcome the challenge of noisy signals, UDI captures how likely a user connects to a signal with respect to 1) the distance between the user and the signal, and 2) the influence scope of the signal. Based on the model, we develop local and global location prediction methods. The experiments on a large scale data set show that our methods improve the state-of-the-art methods by 13%, and achieve the best performance."

default_text_list = [default_text_1, default_text_2, default_text_3, default_text_4]
# # Input 
# default_text = default_text_1 
# user_input = st.text_area("Input your text (paper abstract)", default_text)


# # Generate button
# if (st.button('Generate!',key='b1')):
#     if len(user_input) == 0:
#         st.write('Cannot generate because input is empty.')
#     else:
#         keyword_score_dict, result_keywords, wc_array = keywords(user_input)
#         st.image(wc_array)

#         st.write("Keywords extracted: ",result_keywords)

researcher_name = st.text_input("Name:",'Average Joe')

researcher_affliation = st.text_input("Affliation:", 'University of Kelaideng')

ratio = st.sidebar.slider('What is the ratio of keywords to keep? (%)', 0, 100, 60)

# Generate From Name
if (st.button('Profile!',key='b2')):
    st.write('Info read:')
    st.write(researcher_name,',', researcher_affliation)
    st.write(ratio, '%')
    final_keyword_score_dict = keywords_multiple(default_text_list)

    wc_array, key_list = generate_word_cloud_with_ratio(final_keyword_score_dict, ratio/100)
    st.image(wc_array, use_column_width=True)
    st.write("Keywords extracted: ",key_list)