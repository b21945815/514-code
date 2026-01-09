INSTALL THE DATA FROM https://bird-bench.github.io
IN YOUR PROJECT UNZIP THE DATA TO A FOLDER NAMED DATA
DATA
 --dev_20240627
 --train

make .env file
set GROQ_API_KEY=key

run router_model.py to train router model
run vector_db_builder.py
run app.py with streamlit

BASELINE RESULTS COME FROM
https://github.com/b21945815/Few-shot-NL2SQL-with-prompting
https://github.com/b21945815/DAIL-SQL
They updated to run with with gpt-4o (for financial dataset)
especially for DAIL-SQL the  script or code need more update to provide more correct examples with gpt4-o
