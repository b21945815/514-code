import sys
import os
 
sys.path.append(os.getcwd())
from groq import Groq
from dotenv import load_dotenv
import json
from onePassLlmModel.templates import DECOMPOSITION_PROMPT_TEMPLATE, DECOMPOSITION_PROMPT_WITH_HINT_TEMPLATE
load_dotenv()

class GroqQueryDecomposer:
    """
    Decomposes Natural Language Queries into Structural and Semantic components.
    """
    

    def __init__(self, info_path='info/database_info.json'):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "openai/gpt-oss-120b"
        self.temperature = 0
        
        with open(info_path, 'r', encoding='utf-8') as f:
            self.db_info = json.load(f)

    def decompose_query(self, db_id, user_query, hint=None):
        """
        Main function to call from other scripts.
        Returns a dictionary containing decomposed parts.
        """
        db_meta = self.db_info.get(db_id)
        if not db_meta:
            return {"tasks": [{"is_achievable": False, "error": f"DB {db_id} not found"}]}
        if hint and hint.strip():
            full_prompt = DECOMPOSITION_PROMPT_WITH_HINT_TEMPLATE.format(
                db_metadata=json.dumps(db_meta, indent=2),
                hint=hint
            )
        else:
            full_prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(
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
            return {"tasks": [{"is_achievable": False, "error": str(e)}]}
