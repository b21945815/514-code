import sys
import os
 
sys.path.append(os.getcwd())
import time
from onePassLlmModel.router_model_helper import load_router, predict_intent
from onePassLlmModel.groq_ai_engine import GroqQueryDecomposer
from onePassLlmModel.gpt_ai_engine import GptQueryDecomposer
from onePassLlmModel.sql_compiler import JSONToSQLCompiler
from bird_evaluator import BirdEvaluator 

class BirdSQLPipeline:
    def __init__(self, 
                 model="groq",
                 router_path="./my_router_model", 
                 db_info_path="info/database_info.json", 
                 db_path="financial.sqlite",
                 log_file="evaluation_results.json"):
        
        print("Initializing BirdSQL Pipeline...")
        
        self.router_tokenizer, self.router_model = load_router(router_path)
        print("Router Model Loaded")
        if model == "gpt":
            print("asking to gpt")
            self.decomposer = GptQueryDecomposer(info_path=db_info_path)
        else: 
            self.decomposer = GroqQueryDecomposer(info_path=db_info_path)
        print("Decomposer Engine Loaded")

        self.evaluator = BirdEvaluator(db_filename=db_path)
        print("Evaluator Ready")


    def process_query(self, user_query, db_id="financial", ground_truth_sql=None, hint=None):
        start_time = time.time()
        
        result = {
            "query": user_query,
            "ground_truth_sql": ground_truth_sql,
            "status": "processing",
            "steps": {},
            "metrics": {
                "total_time": 0,
                "total_tokens": 0
            }
        }

        step_router = {"status": "skipped", "intent": None, "confidence": 0.0}
        if self.router_model:
            try:
                intent, score = predict_intent(user_query, self.router_tokenizer, self.router_model)
                step_router = {"status": "success", "intent": intent, "confidence": score}
                if intent == "GENERAL CHAT":
                    result["status"] = "filtered_by_router"
                    result["steps"]["router"] = step_router
                    result["metrics"]["total_time"] = time.time() - start_time
                    return result
            except Exception as e:
                step_router = {"status": "error", "error": str(e)}
        
        result["steps"]["router"] = step_router

        step_decomposer = {"status": "pending", "tokens": 0, "json_plan": None}
        if self.decomposer:
            try:
                json_response, tokens = self.decomposer.decompose_query(db_id, user_query, hint)
                step_decomposer["tokens"] = tokens
                step_decomposer["json_plan"] = json_response
                result["metrics"]["total_tokens"] += tokens
                
                tasks = json_response.get("tasks", [])
                if not tasks:
                    step_decomposer["status"] = "failed_no_tasks"
                elif not tasks[0].get("is_achievable", True):
                    step_decomposer["status"] = "unachievable"
                    step_decomposer["error"] = tasks[0].get("error")
                else:
                    step_decomposer["status"] = "success"
            except Exception as e:
                step_decomposer["status"] = "error"
                step_decomposer["error"] = str(e)
        
        result["steps"]["decomposer"] = step_decomposer

        if step_decomposer["status"] != "success":
            result["status"] = "decomposer_failure"
            result["metrics"]["total_time"] = time.time() - start_time
            return result

        step_compiler = {"status": "pending", "generated_sql": None}
        try:
            compiler = JSONToSQLCompiler(step_decomposer["json_plan"])
            generated_sql = compiler.compile()
            step_compiler["generated_sql"] = generated_sql
            
            if generated_sql.startswith("--"): 
                step_compiler["status"] = "error_in_sql"
            else:
                step_compiler["status"] = "success"
        except Exception as e:
            step_compiler["status"] = "error"
            step_compiler["error"] = str(e)
        
        result["steps"]["compiler"] = step_compiler

        if step_compiler["status"] != "success":
            result["status"] = "compiler_failure"
            result["metrics"]["total_time"] = time.time() - start_time
            return result

        step_evaluator = {"status": "pending", "match_type": "N/A"}
        
        if self.evaluator and ground_truth_sql:
            try:
                token_stats = {"total_tokens": result["metrics"]["total_tokens"]}
                
                eval_result = self.evaluator.evaluate_query(
                    query_id=hash(user_query),
                    natural_language_query=user_query,
                    gt_sql=ground_truth_sql,
                    pred_sql=step_compiler["generated_sql"],
                    token_stats=token_stats
                )
                
                step_evaluator["status"] = "success"
                step_evaluator["match_type"] = eval_result.get("match_type", "WRONG")
                step_evaluator["details"] = eval_result.get("failure_reason")
                step_evaluator["execution_status"] = eval_result.get("execution_status")
                
            except Exception as e:
                step_evaluator["status"] = "error"
                step_evaluator["error"] = str(e)
        elif not ground_truth_sql:
            step_evaluator["status"] = "skipped_no_gt"
        
        result["steps"]["evaluator"] = step_evaluator
        result["status"] = "completed"
        result["metrics"]["total_time"] = time.time() - start_time
        
        return result