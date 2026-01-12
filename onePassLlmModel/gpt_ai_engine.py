import sys
import os
 
sys.path.append(os.getcwd())
from openai import OpenAI
from dotenv import load_dotenv
import json
from onePassLlmModel.templates import DECOMPOSITION_OPEN_AI_PROMPT_TEMPLATE 
load_dotenv()

class GptQueryDecomposer:
    """
    Decomposes Natural Language Queries into Structural and Semantic components.
    """
    
    def __init__(self, info_path='info/database_info.json'):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"
        self.temperature = 0
        
        with open(info_path, 'r', encoding='utf-8') as f:
            self.db_info = json.load(f)

    # Hint is not implemented for gpt
    def decompose_query(self, db_id, user_query, hint=None):
        """
        Main function to call from other scripts.
        Returns a dictionary containing decomposed parts.
        """
        db_meta = self.db_info.get(db_id)
        if not db_meta:
            return {"tasks": [{"is_achievable": False, "error": f"DB {db_id} not found"}]}, 0
        
        full_prompt = DECOMPOSITION_OPEN_AI_PROMPT_TEMPLATE.format(
            db_metadata=json.dumps(db_meta, indent=2)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            total_token = response.usage.total_tokens
            return json.loads(content), total_token
        
        except Exception as e:
            print(e)
            return {"tasks": [{"is_achievable": False, "error": str(e)}]}, 0