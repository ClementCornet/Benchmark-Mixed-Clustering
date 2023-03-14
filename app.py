import streamlit as st
from streamlit_option_menu import option_menu

from streamlit_webapp import compare_dimr, compare_clustering, visualize_dimr

st.set_page_config(page_title="Compare Mixed Clustering",layout="wide")

st.info("TODO SCALING AVANT EMBEDDING \n ET TEST FAMD SCALE INERTIA")
with st.sidebar:
    choice = option_menu("Page",
                    [
                        "Land Page",
                        "Compare Dimension Reductions",
                        "Visualize Dimension Reductions",
                        "Compare Clustering"
                    ])


if choice == "Compare Dimension Reductions":
    compare_dimr.page()
if choice == "Compare Clustering":
    compare_clustering.page_2()
if choice == "Visualize Dimension Reductions":
    visualize_dimr.page()