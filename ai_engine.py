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
You are a SQLLite Decomposition Expert that uses reason. When decomposition check to query to see if your output really returning the asked data with correct filters.

You are strictly faithful to this database schema:
{db_metadata}

Construct a structured JSON object based on the definitions below:

DATA STRUCTURE DEFINITIONS:

1. **ValueNode**: The atomic unit of data
   - LITERAL:    {{ "type": "LITERAL", "value": 100 or "2023-01-01" }}
   - COLUMN:     {{ "type": "COLUMN", "value": "t.amount" }}
   - FUNCTION:   {{ "type": "FUNCTION", "name": "SUM", "params": [ValueNode, ...] }}
   - CASE:       {{ "type": "CASE", "cases": [ {{ "when": ValueNode, "then": ValueNode }} ], "else": ValueNode }}
   - CONDITION:  {{ "type": "CONDITION", "value": ConditionNode }} 
     (Use CONDITION type for Math, Boolean Logic etc. inside values)
   - SEMANTIC:   {{ "type": "SEMANTIC", "value": "rich", "column": "desc", "table": "dist" }}
   - SUBQUERY:   {{ "type": "SUBQUERY", "target_task_id": "2" }}
    (Note: 'target_task_id' is the ID of the other task to be combined with the current task via valueNode)

2. **ConditionNode**: Logic tree. Used in "where_clause", "having_clause", "join_condition" or inside ValueNodes.
   - LEAF:   {{ "left": ValueNode, "operator": "=", "right": ValueNode }}
   - BRANCH: {{ "logic": "AND" | "OR" || "IFNULL" || "IFF" etc., "conditions": [ConditionNode, ConditionNode] }}

3. **StructuralLogic**: Items in the "structural_logic" list.
   - JOIN:       {{ "type": "INNER JOIN", "table": "table as b", "condition": ConditionNode }}
   - SET_OP:     {{ "type": "UNION" | "INTERSECT", "target_task_id": 3 }}
   (Note: 'target_task_id' is the ID of the other task to be combined with the current task via Set Operation)

4. **SelectNode**: Items in the "target" list.
   - {{ "value": ValueNode, "alias": "column_alias" }} 

5. **OrderNode**: Items in the "order_by" list.
   - {{ "value": ValueNode, "direction": "ASC" | "DESC" }}

--- TASK SCHEMA EXPLANATION ---
Each task object represents a single SQL query unit.
- **task_id**: Unique integer ID.
- **main_table**: The primary table in FROM clause (e.g., "orders as o").
- **is_achievable**: Boolean, false if query cannot be answered with schema.
- **target**: List of SelectNodes (The columns to return).
- **use_distinct**: Boolean, true if 'SELECT DISTINCT' is needed.
- **structural_logic**: List of JOINs or SET OPERATIONS.
- **where_clause**: ConditionNode for filtering.
- **group_by**: List of strings (column names) for aggregation.
- **having_clause**: ConditionNode for filtering aggregates.
- **limit_by**: Integer, for LIMIT clause
- **tasks**: array of task objects

--- RULES ---
1. **Semantic Values**: Use SEMANTIC type for the fields in semantic_search field from the database schema.
2. **Aggregates**: Use FUNCTION type for aggregates. Example: SUM(amount) -> {{ "type": "FUNCTION", "name": "SUM", "params": [...] }}
3. **Intent Analysis**: If the query implies an activity (trading, moving money), prioritize activity tables over entity tables.
4. **Semantic Alignment**: Ensure the selected table's description semantically supports the specific action verbs used in the query.
5. **Checking output**: After decomposition check to query to see if your sql logic is correct for asked query and also check if it is can work on SQLLite server. Fix if you found errors

OUTPUT FORMAT EXAMPLE :
{{
  "tasks": [
    {{
      "task_id": 1,
      "is_achievable": true,
      "error": null,
      "limit_by": 10,
      "main_table": "orders as o",
      "is_distinct": true,
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
            {{ "left": {{ "type": "SUBQUERY", "target_task_id": 2 }}, "operator": "IN", "right": {{ "type": "SEMANTIC", "value": "premium", "table": "users", "column": "type" }} }}
         ]
      }},
      "target": [
         {{ "value": {{ "type": "COLUMN", "value": "u.username" }}, "alias": "user" }},
         {{
            "alias": "status",
            "value": {{
               "type": "CASE",
               "cases": [
                  {{ 
                     "when": {{ "type": "CONDITION", "value": {{"left": {{ "type": "COLUMN", "value": "o.amount" }}, "operator": ">", "right": {{ "type": "LITERAL", "value": 1000 }} }} }},
                     "then": {{ "type": "LITERAL", "value": "VIP" }}
                  }}
               ],
               "else": {{ "type": "LITERAL", "value": "Regular" }}
            }}
         }}
      ],
      "group_by": ["u.username"],
      "having_clause": {{
          "left": {{ "type": "FUNCTION", "name": "COUNT", "params": [{{ "type": "COLUMN", "value": "o.id" }}] }}, 
          "operator": ">", 
          "right": {{ "type": "LITERAL", "value": 5 }} 
      }},
      "order_by": [
         {{ "value": {{ "type": "COLUMN", "value": "taxed_amount" }}, "direction": "DESC" }}
      ],
      "subqueries": {{
         "@avg_order": {{
             "main_table": "orders",
             "target": [ {{ "value": {{ "type": "EXPRESSION", "value": "AVG(total_amount)" }}, "alias": null }} ]
         }}
      }}
    }},
    {{
      "task_id": 2,
      "main_table": "orders",
      "target": [
         {{ 
            "value": {{ "type": "FUNCTION", "name": "AVG", "params": [{{ "type": "COLUMN", "value": "total_amount" }}] }}, 
            "alias": null 
         }}
      ]
    }}
  ]
}}
"""

    def __init__(self, info_path='info/database_info.json'):
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
            total_token = response.usage.total_tokens
            return json.loads(content), total_token
        
        except Exception as e:
            return {"tasks": [{"is_achievable": False, "error": str(e)}]}
