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
You are a SQL Decomposition Expert that uses reason. When decomposition check to query to see if your output really returning the asked data with correct filters.

You are strictly faithful to this database schema:
{db_metadata}

Construct a structured JSON object based on the definitions below:

DATA STRUCTURE DEFINITIONS:

1. **ValueNode**: Represents data units or logic blocks.
   - LITERAL:    {{ "type": "LITERAL", "value": 100 or "2023-01-01" }}
   - SEMANTIC:   {{ "type": "SEMANTIC", "value": "rich districts" }}
   - COLUMN:     {{ "type": "COLUMN", "value": "SUM(t.amount)" }}
   - COLUMN:     {{ "type": "COLUMN", "value": "t.amount" }}
   - SUBQUERY:   {{ "type": "SUBQUERY", "value": "@sub1" }}
   - EXPRESSION: {{ "type": "EXPRESSION", "value": "CASE WHEN t.amount > 0 THEN 1 ELSE 0 END" or "SUM(t.amount)" }} 
     (Use EXPRESSION for Math, Functions, and CASE logic)

2. **SelectNode**: Items in the "target" list.
   - {{ "value": ValueNode, "alias": "column_alias" }} 

3. **OrderNode**: Items in the "order_by" list.
   - {{ "value": ValueNode, "direction": "ASC" | "DESC" }}

4. **ConditionNode**: Recursive logic for "where_clause" and "having_clause".
   - LEAF:   {{ "left": ValueNode, "operator": "=", "right": ValueNode }}
   - BRANCH: {{ "logic": "AND" | "OR" || "IFNULL" || "IFF" etc., "conditions": [ConditionNode, ConditionNode] }}

5. **LogicNode**: Elements inside "structural_logic".
   - JOIN:   {{ "type": "INNER JOIN", "table": "account as a", "condition": ConditionNode }}
   - SET_OP: {{ "type": "UNION" | "INTERSECT", "target_task_id": 2 }}

RULES:
1 - Use SEMANTIC type for ValueNode instead of LITERAL when the table column is semantic (Not a number and not a date)
2 - If you can not generate valid output for a given query set is_achievable to false and explain the reason in error 
3 - Intent Analysis: Distinguish between 'Entity Definition' and 'Entity Activity' If the query implies an activity, prioritize activity tables over entity tables.
4 - Semantic Alignment: Ensure the selected table's description semantically supports the specific action verbs used in the query
OUTPUT FORMAT EXAMPLE 
{{
  "tasks": [
    {{
      "task_id": 1,
      "is_achievable": true,
      "error": null,
      "limit_by": 10,
      "main_table": "orders as o",
      "structural_logic": [
        {{ 
           "type": "INNER JOIN", 
           "table": "users as u", 
           "condition": {{ "left": {{ "type": "COLUMN", "value": "o.user_id" }}, "operator": "=", "right": {{ "type": "COLUMN", "value": "u.id" }} }} 
        }}
      ],
      "where_clause": {{
         "logic": "AND",
         "conditions": [
            {{ "left": {{ "type": "COLUMN", "value": "u.userType" }}, "operator": "IN", "right": {{ "type": "SEMANTIC", "value": "premium account" }},
            {{ "left": {{ "type": "COLUMN", "value": "o.total_amount" }}, "operator": ">", "right": {{ "type": "SUBQUERY", "value": "@avg_order" }} }}
         ]
      }},
        "target": {{
            "targets"[
                {{ "value": {{ "type": "COLUMN", "value": "u.username" }}, "alias": "user" }},
                {{ "value": {{ "type": "EXPRESSION", "value": "CASE WHEN SUM(o.total_amount) > 5000 THEN 'VIP' ELSE 'Regular' END" }}, "alias": "status" }}
            ]
        "use_distinct": true
      }},
      "group_by": ["u.username"],
      "having_clause": {{
         "logic": "AND",
         "conditions": [
             {{ "left": {{ "type": "EXPRESSION", "value": "COUNT(o.id)" }}, "operator": ">", "right": {{ "type": "LITERAL", "value": 5 }} }}
         ]
      }},
      "order_by": [
         {{ "value": {{ "type": "EXPRESSION", "value": "status" }}, "direction": "DESC" }}
      ],
      "subqueries": {{
         "@avg_order": {{
             "main_table": "orders",
             "target": [ {{ "value": {{ "type": "EXPRESSION", "value": "AVG(total_amount)" }}, "alias": null }} ]
         }}
      }}
    }}
  ]
}}
"""

    def __init__(self, info_path='database_info.json'):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "openai/gpt-oss-120b"
        self.temperature =0.75
        
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
        "Order pizza for me.", # Should trigger is_achievable: False,
        "List the IDs of clients who live in Prague and union them with the IDs of clients who have a gold card.", # Union,
        "Name the account numbers of female clients who are oldest and have lowest average salary?", # moderate
        "List out the accounts who have the earliest trading date in 1995 ?", # simple
        "For the branch which located in the south Bohemia with biggest number of inhabitants, what is the percentage of the male clients?" # challenging
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