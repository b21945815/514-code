INSTALL THE DATA FROM https://bird-bench.github.io
IN YOUR PROJECT UNZIP THE DATA TO A FOLDER NAMED DATA
DATA
 --dev_20240627
 --train

update your environment via environment.yml file

make .env file
set GROQ_API_KEY=key
OPENAI_API_KEY=key

run router_model.py to train router model (train data only needed here, it is very big so we could not share, you can use test data as well via updating the code)
row 147 148 should be uncommented if you have train data

run vector_db_builder.py to build vector database records

run app.py with "streamlit run app.py" command


reprocess_results.py is for testing changes for sql_compiler without asking AI again

test_system.py run our testing

info/database_info.json used in LLM script

database_info_mappings.json used for vectorDB

