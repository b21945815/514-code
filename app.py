import streamlit as st
from onePassLlmModel.router_model_helper import load_router, predict_intent
from onePassLlmModel.groq_ai_engine import GroqQueryDecomposer
from onePassLlmModel.gpt_ai_engine import GptQueryDecomposer
from onePassLlmModel.sql_compiler import JSONToSQLCompiler

st.set_page_config(page_title="BirdSQL Execution Plan", layout="wide")

@st.cache_resource
def init_models():
    tokenizer, model = load_router("./my_router_model")
    decomposerGROQ = GroqQueryDecomposer("info/database_info.json")
    decomposerGPT = GptQueryDecomposer("info/database_info.json")
    return tokenizer, model, decomposerGPT, decomposerGROQ

tokenizer, model, decomposerGPT, decomposerGROQ = init_models()

st.sidebar.title("🗄️ Control Panel")
selected_db = st.sidebar.selectbox("Active Model", ["GROQ", "GPT"])

st.title("BirdSQL Pipeline Visualizer")

user_input = st.text_input("Enter your query:", placeholder="e.g., Get loans in Prague UNION get clients with gold cards")

if user_input:
    intent, score = predict_intent(user_input, tokenizer, model)

    if intent == "GENERAL CHAT":
        with st.chat_message("assistant"):
            st.write(f"**Intent Detected:** General Chat ({score*100:.1f}%)")
            st.write("""
            I am the **Natural Query to SQL Assistant**. My specialized job is to:
            1. **Analyze** your natural language questions about the database.
            2. **Decompose** complex requests into logical execution steps (Tasks).
            3. **Bridge** the gap between human language and SQL using a Hybrid approach.
            
            Please ask me something about the selected database!
            """)
    else:
        with st.spinner("Generating Logical Execution Plan..."):
            if selected_db == "GPT":
                print(111)
                response, total_tokens = decomposerGPT.decompose_query("financial", user_input, None)
            else:
                response, total_tokens = decomposerGROQ.decompose_query("financial", user_input, None)
            tasks = response.get("tasks", [])
            
            st.divider()
            col_info1, col_info2 = st.columns([3, 1])
            with col_info1:
                st.success(f"**Analysis Complete:** SQL Intent Detected ({score*100:.1f}%)")
            with col_info2:
                st.metric(label="LLM Tokens Used", value=total_tokens, help="Total input+output tokens consumed by the Decomposer LLM")
            st.divider()

            if not tasks:
                st.warning("Decomposer returned no tasks for this query.")
            else:
                has_error = any(not task.get("is_achievable", True) for task in tasks)
                compiler = JSONToSQLCompiler(response)

                if has_error:
                    st.subheader("Execution Plan (Errors Detected)")
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
                                target_id = op.get("target_task_id")
                                op_name = op.get("type")
                                st.markdown(f"""
                                    <div style="border-left: 5px solid #ff4b4b; padding-left: 20px; margin: 10px 0;">
                                        <h3 style="color: #ff4b4b;">⚡ {op_name}</h3>
                                        <p style="opacity: 0.7;">Combine results with <b>Task ID: {target_id}</b></p>
                                    </div>
                                """, unsafe_allow_html=True)
                else:
                    st.subheader("Final Generated SQL")
                    try:
                        final_sql = compiler.compile()
                        st.code(final_sql, language="sql")                     
                    except Exception as e:
                        st.error(f"Compilation Error: {e}")