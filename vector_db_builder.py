import sqlite3
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

from pyparsing import col

class VectorDBBuilder:
    def __init__(self, db_path, info_path, db_id='financial'):
        self.db_path = db_path
        self.db_id = db_id
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        info_path = os.path.join(BASE_DIR, 'info', info_path)
        info_path = os.path.abspath(info_path)
        # Load Metadata
        with open(info_path, 'r', encoding='utf-8') as f:
            self.full_info = json.load(f)
            self.db_info = self.full_info[db_id]
            
        # Initialize ChromaDB (Persistent)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Embedding Function (using a lightweight model)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

    def build_collections(self):
        semantic_cols = self.db_info.get('text_columns', [])
        print(f"Semantic Columns to process: {semantic_cols}")

        mappings = self.db_info.get('value_mappings', {})
        
        conn = sqlite3.connect(self.db_path)
        
        print(f"--- Starting Vectorization for {len(semantic_cols)} columns ---")
        
        for full_col_name in semantic_cols:
            table, col = full_col_name.split('.')
            print(f"\nProcessing: {full_col_name}")
            
            # 1. Make/Get Collection
            collection_name = f"{table}_{col}"
            try:
                self.chroma_client.delete_collection(name=collection_name)
                print(f"   -> Deleted existing collection '{collection_name}'")
            except:
                pass
            
            collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.emb_fn
            )
            
            # 2. Strategy: Check if Mapping exists (e.g., k_symbol -> "insurance payment")
            # Mappings can be namespaced (loan.status) or generic (k_symbol)
            mapping_data = mappings.get(full_col_name) or mappings.get(col)
            print(f"   -> Found Mapping: {bool(mapping_data)}")
            print("mapping_data:", mapping_data)
            documents = [] # What we embed (English description or Raw Text)
            metadatas = [] # The actual DB value to return
            ids = []
            
            if mapping_data and isinstance(mapping_data, dict):
                print(f"   -> Using MAPPINGS (Semantic Translation)")
                # Example: key="POJISTNE", value="insurance payment"
                for db_val, description in mapping_data.items():
                    documents.append(description)  # Embed "insurance payment"
                    metadatas.append({"db_value": db_val, "original": description})
                    ids.append(f"{db_val}")
                    
            else:
                print(f"   -> Using RAW DB VALUES (Distinct lookup)")
                # Example: district.A2 -> "Prague", "Brno"
                # Escape table and column names with double quotes to handle keywords like "order"
                query = f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL'
                print(f"     Executing Query: {query}")
                df = pd.read_sql_query(query, conn)
                
                unique_vals = df[col].astype(str).tolist()
                for val in unique_vals:
                    if not val or not val.strip():
                        continue
                    documents.append(val) # Embed "Prague"
                    metadatas.append({"db_value": val, "original": val})
                    ids.append(f"{val}")

            # 3. Add to Chroma
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"   -> Stored {len(documents)} items in collection '{collection_name}'")
            else:
                print("   -> No data found!")

        conn.close()
        print("\n Vector Database Built Successfully in ./chroma_db")

    def list_collections(self):
        collections = self.chroma_client.list_collections()
        print("--- Existing Collections ---")
        for col in collections:
            print(f" - {col.name}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR, 'data', 'dev_20240627', 'dev_databases', 'financial', 'financial.sqlite')
    DB_PATH = os.path.abspath(DB_PATH)
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB file could not find in:\n{DB_PATH}")
        print("please try 'find . -name financial.sqlite' in terminal and paste here in the code.")
        exit(1)
        
    builder = VectorDBBuilder(
        db_path=DB_PATH,
        info_path='database_info_mappings.json'
    )
    builder.build_collections()
    builder.list_collections()