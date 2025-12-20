import streamlit as st
from router_model_helper import load_router, predict_intent
from query_decomposer import QueryDecomposer

st.set_page_config(page_title="BirdSQL Execution Plan", layout="wide")

@st.cache_resource
def init_models():
    tokenizer, model = load_router("./my_router_model")
    decomposer = QueryDecomposer("database_info.json")
    return tokenizer, model, decomposer

tokenizer, model, decomposer = init_models()

# Sidebar
st.sidebar.title("🗄️ Control Panel")
selected_db = st.sidebar.selectbox("Active Database", ["financial"])

st.title("BirdSQL Pipeline Visualizer")

user_input = st.text_input("Enter your complex query:", placeholder="e.g., Get loans in Prague UNION get clients with gold cards")

if user_input:
    intent, score = predict_intent(user_input, tokenizer, model)

    if intent == "GENERAL CHAT":
        with st.chat_message("assistant"):
            st.write(f"**Intent Detected:** General Chat ({score*100:.1f}%)")
            st.write("""
            I am the **Natural Wuery to SQL Assistant**. My specialized job is to:
            1. **Analyze** your natural language questions about the database.
            2. **Decompose** complex requests into logical execution steps (Tasks).
            3. **Bridge** the gap between human language and SQL using a Hybrid approach.
            
            Please ask me something about the selected database!
            """)
    else:
        with st.spinner("Generating Logical Execution Plan..."):
            response = decomposer.decompose_query(selected_db, user_input)
            tasks = response.get("tasks", [])

            if not tasks:
                st.warning("Decomposer returned no tasks for this query.")
            else:
                st.subheader("Step-by-Step Execution Plan")
                
                for i, task in enumerate(tasks):
                    t_id = task.get("task_id")
                    with st.expander(f"STEP {i+1} | Task ID: {t_id}", expanded=True):
                        if not task.get("is_achievable", True):
                            st.error(f"Logical Error: {task.get('error')}")
                            st.json(task) 
                        else:
                            st.json(task)                    

                    structural_logic = task.get("structural_logic", [])
                
                    set_ops = [l for l in structural_logic if l.get("type") in ["UNION", "INTERSECT", "EXCEPT"]]
                    
                    if set_ops:
                        for op in set_ops:
                            target_id = op.get("task_id")
                            op_name = op.get("type")
                            st.markdown(f"""
                                <div style="border-left: 5px solid #ff4b4b; padding-left: 20px; margin: 10px 0;">
                                    <h3 style="color: #ff4b4b;">⚡ {op_name}</h3>
                                    <p style="opacity: 0.7;">Combine results with <b>Task ID: {target_id}</b></p>
                                </div>
                            """, unsafe_allow_html=True)
                    