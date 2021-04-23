import streamlit as st
from MAG import MAG_get_abstracts
from visualization import generate_word_cloud
import time
from profiling import Profile

# Get the name and affi for profiling
st.markdown('# Keyword Extraction based on EmbedRank:')
researcher_affiliation = st.text_input('Affiliation: ', 'University of Illinois at Urbana Champaign')
researcher_name = st.text_input('Name: ','Kevin Chenchuan Chang')

# get user-specified parameters
time_ratio_num = st.sidebar.slider('What is the ratio of time span to keep? (%)', 0, 100, (0,100))
time_ratio = (time_ratio_num[0]/100, time_ratio_num[1]/100)
keyword_ratio_num = st.sidebar.slider('What is the ratio of keywords to keep? (%)', 0, 100, 90)
keyword_ratio = (keyword_ratio_num/100)**2.5
diversity_num = st.sidebar.slider('How diverse do we want the keywords to be? 100 being most diverse (%)', 0, 100, 20)
diversity = diversity_num/100
# output the parameters got for user to confirm 
st.text('The ratio of time span you selected is: {}% to {}% of their whole career'.format(time_ratio_num[0], time_ratio_num[1]))
st.text('The ratio of keywords you selected is: {}%'.format(keyword_ratio_num))
st.text('The diversity you selected is: {}%'.format(diversity_num))
profile = Profile(researcher_name, researcher_affiliation, diversity=diversity)

# Generate From Name
if (st.button('Profile!',key='b2')):
    init = st.text('Start profiling.')
    init.text('Start profiling, querying MAG...')
    profile.get_abstracts_from_MAG()
    init.text('MAG querying finished.')
    time.sleep(1)
    profile.filter_abstracts(time_ratio=(time_ratio))
    init.text('The corresponding time span is from year {} to {}.'.format(profile.starting_year, profile.ending_year)) 
    if len(profile.abstracts) != 0:
        final_keyword_score_dict = profile.extract_keywords(keyword_ratio=keyword_ratio)
        _, wc_array, key_list = generate_word_cloud(final_keyword_score_dict)
        st.image(wc_array, use_column_width=True)
        st.write("Keywords extracted: ", key_list)
        subfield_score = profile.evaluate_subfields()
        st.write(dict(zip(profile.basis_words, subfield_score)))

        # st.write(len(profile.basis_words))
        # st.write(len(subfield_score))
    else:
        st.write("No abstracts found for that researcher!")


# TODO: 1. lemmatizing the keywords (graphs->graph) 2. scale the keywords by the csv