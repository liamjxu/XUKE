import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keywords import keywords

st.markdown("# Keyword Extraction based on TextRank:")

# Input textbox
user_input = st.text_area("Input your text (paper abstract)")

# Generate button
if (st.button('Generate!',key='b1')):
    if len(user_input) == 0:
        st.write('Cannot generate because input is empty.')
    else:
        keyword_score_dict, result_keywords, wc_array = keywords(user_input)
        st.image(wc_array)

        st.write("Keywords extracted: ",keyword_score_dict)
