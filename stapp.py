import streamlit as st
from keywords import keywords

st.markdown("# Keyword Extraction based on TextRank:")

default_text_1 = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."

default_res_name = 'Average Joe'
default_res_affliation = 'University of Clayton'

# # Input 
default_text = default_text_1 
user_input = st.text_area("Input your text (paper abstract)", default_text)


# Generate button
if (st.button('Generate!',key='b1')):
    if len(user_input) == 0:
        st.write('Cannot generate because input is empty.')
    else:
        keyword_score_dict, result_keywords, wc_array = keywords(user_input)
        st.image(wc_array)

        st.write("Keywords extracted: ",result_keywords)

researcher_name = st.text_input("Name:",default_res_name)
researcher_affliation = st.text_input("Affliation:", default_res_affliation)
ratio = st.slider('What is the ratio of keywords to keep? (%)', 0, 100, 60)

# Generate From Name
if (st.button('Go with name!',key='b2')):
    st.write('Info read:')
    st.write(researcher_name,',', researcher_affliation)
    st.write(ratio, '%')

st.write(ratio)