import streamlit as st
from keywords import keywords_multiple
from MAG import MAG_get_abstracts
from visualization import generate_word_cloud

st.markdown("# Keyword Extraction based on TextRank:")
researcher_affliation = st.text_input("Affliation:", 'University of Illinois at Urbana Champaign')
researcher_name = st.text_input("Name:",'Kevin Chenchuan Chang')
ratio_num = st.sidebar.slider('What is the ratio of keywords to keep? (%)', 0, 100, 50)
ratio = (ratio_num/100)**2.5

# Generate From Name
if (st.button('Profile!',key='b2')):
    st.write('Info read:')
    st.write(researcher_affliation, ',', researcher_name)
    st.write(ratio_num, '%')

    abstract_list = MAG_get_abstracts(researcher_affliation, researcher_name)
    final_keyword_score_dict = keywords_multiple(abstract_list, ratio)
    wc_array, key_list = generate_word_cloud(final_keyword_score_dict, ratio)
    st.image(wc_array, use_column_width=True)
    st.write("Keywords extracted: ", key_list)