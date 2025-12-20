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
You are a SQL Decomposition Expert. Break down the user's natural language request into a JSON object containing a LIST of "tasks".

DATABASE SCHEMA AND METADATA:
{db_metadata}

INSTRUCTIONS:
1. "tasks": A list of independent query blocks. If the user asks for two unrelated things, provide two separate task objects.
2. "subqueries": A dictionary within a task where keys are aliases (e.g., "@sub1"). Values MUST be full query objects.
3. "structural_logic": List of objects: {{"type": "JOIN", "table": "table as alias", "condition": "a.id = b.id"}} or {{"type": "UNION/INTERSECT", "task_id": 2}}.
4. "semantic_mappings": Terms for Vector DB lookup (e.g., {{"district.A2": "Prague"}}).
5. "is_achievable": False if the schema cannot answer the question
6. "error": If is_achievable is false, explain why to the user.

OUTPUT FORMAT EXAMPLE:
{{
  "tasks": [
    {{
      "task_id": 1,
      "is_achievable": true,
      "error": null,
      "main_table": "trans as t",
      "intent": "SELECT",
      "limit by": "10",
      "structural_logic": [
        {{ "type": "JOIN", "table": "account as a", "condition": "t.account_id = a.account_id" }}
      ],
      "semantic_mappings": {{"t.k_symbol": "insurance"}},
      "exact_filters": {{"t.balance": "> @sub1"}},
      "group_by": ["t.account_id"],
      "order_by": ["t.account_id desc"],
      "having_filters": ["AVG(t.amount) > 1000"],
      "subqueries": {{
        "@sub1": {{
           "main_table": "trans",
           "intent": "AVG",
           "target": ["balance"]
        }}
      }},
      "target": ["t.account_id", "t.balance"]
    }}
  ]
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
            return {"tasks": [{"is_achievable": False, "error": f"DB {db_id} not found"}]}

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
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        
        except Exception as e:
            return {"tasks": [{"is_achievable": False, "error": str(e)}]}

# --- TESTING SECTION ---
if __name__ == "__main__":
    decomposer = QueryDecomposer()
    
    # Test queries including simple, complex, and multi-task scenarios
    test_queries = [
        "Show me the average balance of female clients from Prague who have gold cards.", # Complex join
        "List all loans in Prague and find the youngest client in Brno.", # Independent Multi-Task
        "Who are the clients with a balance higher than the average balance?", # Subquery
        "Order pizza for me." # Should trigger is_achievable: False,
        "List the IDs of clients who live in Prague and union them with the IDs of clients who have a gold card." # Union
    ]
    
    print(f"{'='*20} TESTING DECOMPOSER {'='*20}\n")
    
    for i, sample_query in enumerate(test_queries, 1):
        print(f"Test {i}: {sample_query}")
        
        result = decomposer.decompose_query("financial", sample_query)
        
        tasks = result.get("tasks", [])
        
        if tasks:
            for task in tasks:
                task_id = task.get("task_id", "N/A")
                if task.get("is_achievable"):
                    print(f"Task {task_id} Decomposed Successfully")
                else:
                    print(f"Task {task_id} Failed: {task.get('error')}")
            
            print("\nFull Response JSON:")
            print(json.dumps(result, indent=4))
        else:
            print("Unexpected response format or empty task list.")
            
        print("-" * 50)