import streamlit as st
from router_model_helper import load_router, predict_intent
from query_decomposer import QueryDecomposer

st.set_page_config(page_title="BirdSQL AI", layout="centered")

@st.cache_resource
def init_models():
    tokenizer, model = load_router("./my_router_model")
    decomposer = QueryDecomposer("database_info.json")
    return tokenizer, model, decomposer

tokenizer, model, decomposer = init_models()

st.sidebar.title("Databases")
selected_db = st.sidebar.selectbox("Active Database", ["financial"])

st.title("🦅 BirdSQL Assistant")

user_input = st.text_input("Enter your question:", placeholder="e.g., Show gold card holders in Prague")

if user_input:
    intent, score = predict_intent(user_input, tokenizer, model)

    if intent == "GENERAL CHAT":
        st.info(f"Intent: {intent} (Confidence: {score*100:.1f}%)")
        st.write("I am a just a native sql helper. Please ask a database-related question.")
    
    else:
        st.success(f"Intent: DATABASE QUERY (Confidence: {score*100:.1f}%)")
        
        with st.spinner("Decomposing..."):
            res_json = decomposer.decompose_query(selected_db, user_input)
            
            st.subheader("Query Structure")
            st.json(res_json)