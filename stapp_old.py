import streamlit as st
from keywords import keywords_multiple, filter_abstracts
from MAG import MAG_get_abstracts
from visualization import generate_word_cloud
import time
from profile import Profile

# Get the name and affi for profiling
st.markdown('# Keyword Extraction based on TextRank:')
researcher_affliation = st.text_input('Affiliation: ', 'University of Illinois at Urbana Champaign')
researcher_name = st.text_input('Name: ','Kevin Chenchuan Chang')

# get user-specified parameters
time_ratio_num = st.sidebar.slider('What is the ratio of time span to keep? (%)', 0, 100, (67,100))
time_ratio = (time_ratio_num[0]/100, time_ratio_num[1]/100)
keyword_ratio_num = st.sidebar.slider('What is the ratio of keywords to keep? (%)', 0, 100, 50)
keyword_ratio = (keyword_ratio_num/100)**2.5
diversity_num = st.sidebar.slider('How diverse do we want the keywords to be? 100 being most diverse (%)', 0, 100, 90)
diversity = diversity_num/100
# output the parameters got for user to confirm 
st.text('The ratio of time span you selected is: {}% to {}% of their whole career'.format(time_ratio_num[0], time_ratio_num[1]))
st.text('The ratio of keywords you selected is: {}%'.format(keyword_ratio_num))
st.text('The diversity you selected is: {}%'.format(diversity_num))

# Generate From Name
if (st.button('Profile!',key='b2')):
    profile = Profile(researcher_name, researcher_affliation)
    init = st.text('Start profiling, querying MAG...')
    abstract_list, starting_year, ending_year = filter_abstracts(MAG_get_abstracts(researcher_affliation, researcher_name),time_ratio)
    init.text('MAG querying finished.')
    time.sleep(1)
    init.text('The corresponding time span is from year {} to {}.'.format(starting_year, ending_year))
    if len(abstract_list) != 0:
        final_keyword_score_dict = keywords_multiple(abstract_list, keyword_ratio, diversity)
        _, wc_array, key_list = generate_word_cloud(final_keyword_score_dict)
        st.image(wc_array, use_column_width=True)
        st.write("Keywords extracted: ", key_list)
    else:
        st.write("No abstracts found for that researcher!")