import chromadb
from chromadb.utils import embedding_functions

class VectorSearcher:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

    def search(self, table, column, query_text, n_results=1):
        """
        Searches for the closest match in the specified table_column collection.
        Returns: The actual DB value to use in SQL.
        """
        collection_name = f"{table}_{column}"
        
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.emb_fn
            )
            
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            if not results['metadatas'][0]:
                return None, 0.0
            
            # Extract best match
            best_match_meta = results['metadatas'][0][0] # {'db_value': 'POJISTNE', 'original': 'insurance payment'}
            distance = results['distances'][0][0] # Lower is better in Chroma (L2) usually, but check metric
            
            # Simple confidence score simulation (inverse of distance)
            # Adjust logic based on your needs
            confidence = 1 / (1 + distance)
            
            return best_match_meta['db_value'], confidence

        except Exception as e:
            print(f"Vector Search Error ({collection_name}): {e}")
            return None, 0.0

# --- Test ---
if __name__ == "__main__":
    searcher = VectorSearcher()
    
    test_cases = [
        ("district", "A2", "Prague"),       # Should find "Hl.m. Praha"
        ("district", "A3", "lower Moravia"), 
        ("trans", "k_symbol", "pension"),   # Should find "DUCHOD"
        ("trans", "k_symbol", "insurenje"), # Typo test -> Should find "POJISTNE"
        ("loan", "status", "NOT paid yet"), # Should find "B"
        ("client", "gender", "female"), # Should find "B"
        ("client", "gender", "male") # Should find "B"
    ]
    
    print(f"{'Query':<20} | {'Found DB Value':<20} | {'Score':<5}")
    print("-" * 55)
    
    for t, c, q in test_cases:
        val, score = searcher.search(t, c, q)
        print(f"{q:<20} | {str(val):<20} | {score:.2f}")