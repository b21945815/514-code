import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class QueryDecomposer:
    """
    Decomposes Natural Language Queries into Structural and Semantic components.
    """
    
    DECOMPOSITION_PROMPT_TEMPLATE = """
You are a SQL Decomposition Specialist. 
Your task is to analyze a natural language question and break it down for a Hybrid SQL approach.

DATABASE SCHEMA AND METADATA:
{db_metadata}

INSTRUCTIONS:
1. "main_table": Identify the primary table where the target data resides.
2. "structural_logic": List ONLY necessary JOINs. If the target column is in the main_table, do not join others unless filtered or needed for group by etc. .
3. "semantic_search": Terms for Vector DB lookup.
4. "exact_filters": Direct SQL filters (e.g., gender = 'F').
5. "group_by": Columns for aggregation categories (e.g., "per district").
6. "target": The specific columns or aggregations (e.g., ["AVG(trans.amount)"]).

OUTPUT FORMAT:
Return ONLY a JSON object.
Example format:
{{
  "main_table": "table",
  "structural_logic": ["JOIN table with table2"],
  "semantic_search": {{"table.column": ["term"]}},
  "exact_filters": {{"table.column": "value"}},
  "group_by": ["table.column"],
  "intent": "SELECT / COUNT / AVG / SUM",
  "target": ["table.amount"]
}}
"""

    def __init__(self, info_path='database_info.json'):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        
        with open(info_path, 'r', encoding='utf-8') as f:
            self.db_info = json.load(f)

    def decompose_query(self, db_id, user_query):
        """
        Main function to call from other scripts.
        Returns a dictionary containing decomposed parts.
        """
        db_meta = self.db_info.get(db_id)
        if not db_meta:
            return {"error": f"Database {db_id} not found in info file."}

        # Prompt
        full_prompt = self.DECOMPOSITION_PROMPT_TEMPLATE.format(
            db_metadata=json.dumps(db_meta, indent=2)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        
        except Exception as e:
            return {"error": f"Groq API Error: {str(e)}"}

# TEST
if __name__ == "__main__":
    decomposer = QueryDecomposer()
    
    sample_query = "Show me the average balance of female clients from Prague who have gold cards."
    
    print(f"--- TESTING DECOMPOSER ---")
    print(f"Input Query: {sample_query}\n")
    
    result = decomposer.decompose_query("financial", sample_query)
    
    if "error" not in result:
        print("Decomposition Successful:")
        print(json.dumps(result, indent=4))
    else:
        print(f"Failed: {result['error']}")