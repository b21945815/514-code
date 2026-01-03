from datetime import datetime
from bird_db_reader import BirdDBReader 

class BirdEvaluator:
    def __init__(self, db_filename="financial.sqlite"):
        self.db_filename = db_filename
        self.db_reader = BirdDBReader(db_filename)
        
    def _execute_sql(self, sql):
        try:
            with self.db_reader as db:
                df = db.run_select_query(sql)
                
                if df is None:
                    return None, [], "Query returned None"
                
                col_names = [str(col).strip().lower() for col in df.columns]
                
                normalized_rows = []
                for row in df.values:
                    norm_row = [str(item).strip().lower() if item is not None else "none" for item in row]
                    normalized_rows.append(norm_row)
                
                return normalized_rows, col_names, None

        except Exception as e:
            return None, [], str(e)

    def _check_strict_accuracy(self, gt_rows, pred_rows):
        gt_set = set(tuple(r) for r in gt_rows)
        pred_set = set(tuple(r) for r in pred_rows)
        
        return gt_set == pred_set

    def _check_soft_accuracy_ordered_mapped(self, gt_rows, pred_rows):
        if len(pred_rows) < len(gt_rows):
            return False, f"Row Count Mismatch: Expected {len(gt_rows)}, Got {len(pred_rows)}"
        
        if len(gt_rows) == 0:
            return True, "Empty Result Match"

        if len(pred_rows[0]) < len(gt_rows[0]):
            return False, f"Column Count Mismatch: Expected {len(gt_rows[0])}+, Got {len(pred_rows[0])}"
        
        gt_row0 = gt_rows[0]
        pred_row0 = pred_rows[0]
        
        column_map = {} 
        used_pred_indices = set() 

        for gt_idx, gt_val in enumerate(gt_row0):
            found_match = False
            for pred_idx, pred_val in enumerate(pred_row0):
                if pred_idx in used_pred_indices:
                    continue 
                if gt_val == pred_val:
                    column_map[gt_idx] = pred_idx
                    used_pred_indices.add(pred_idx)
                    found_match = True
                    break
            
            if not found_match:
                return False, f"Mapping Failed: GT Column {gt_idx} ({gt_val}) not found in Prediction Row 0."

        for row_idx, gt_row in enumerate(gt_rows):
            pred_row = pred_rows[row_idx]
            
            for gt_col_idx, gt_val in enumerate(gt_row):
                pred_col_idx = column_map[gt_col_idx]
                pred_val = pred_row[pred_col_idx]
                
                if gt_val != pred_val:
                    return False, f"Value Mismatch at Row {row_idx}, GT Col {gt_col_idx}: Expected '{gt_val}', Got '{pred_val}'"

        return True, "Soft Match Successful (Mapped & Ordered)"
    
    def _check_super_soft_accuracy(self, gt_rows, pred_rows):
        if len(pred_rows) < len(gt_rows):
            return False, f"Row Count Mismatch: Expected {len(gt_rows)}, Got {len(pred_rows)}"
        
        if not gt_rows:
            return True, "Empty Result Match"

        used_pred_indices = set()

        for i, gt_row in enumerate(gt_rows):
            gt_row_set = set(gt_row) 
            
            match_found = False
            
            for j, pred_row in enumerate(pred_rows):
                if j in used_pred_indices:
                    continue
                
                pred_row_set = set(pred_row)
                
                if gt_row_set.issubset(pred_row_set):
                    match_found = True
                    used_pred_indices.add(j) 
                    break
            
            if not match_found:
                return False, f"Row {gt_row} (as set) not found in prediction rows (subset check)."

        return True, "Super Soft Match Successful (Rows as Sets & Subset Check)"

    def evaluate_query(self, query_id, natural_language_query, gt_sql, pred_sql, token_stats=None):
        gt_res, gt_cols, gt_err = self._execute_sql(gt_sql)
        pred_res, pred_cols, pred_err = self._execute_sql(pred_sql)

        result_log = {
            "timestamp": datetime.now().isoformat(),
            "query_id": query_id,
            "nl_query": natural_language_query,
            "sql_ground_truth": gt_sql,
            "sql_predicted": pred_sql,
            "token_stats": token_stats or {},
            "execution_status": "SUCCESS",
            "match_type": "NONE",
            "failure_reason": None,
            "result_summary": {
                "gt_rows": len(gt_res) if gt_res else 0,
                "pred_rows": len(pred_res) if pred_res else 0,
                "gt_cols": len(gt_cols),
                "pred_cols": len(pred_cols)
            }
        }

        if pred_err:
            result_log["execution_status"] = "SQL_ERROR"
            result_log["failure_reason"] = f"SQL Execution Failed: {pred_err}"
            return result_log

        if gt_res == pred_res:
            result_log["match_type"] = "EXACT_MATCH"
            return result_log
        
        if self._check_strict_accuracy(gt_res, pred_res):
            result_log["match_type"] = "STRICT_EXACT_MATCH"
            return result_log

        is_soft, soft_msg = self._check_soft_accuracy_ordered_mapped(gt_res, pred_res)
        
        if is_soft:
            result_log["match_type"] = "SOFT_MATCH"
            details = []
            if len(pred_res) > len(gt_res): details.append("Extra Rows")
            if len(pred_cols) > len(gt_cols): details.append("Extra Columns")
            
            reason = f"Logic Correct (Smart Map): {', '.join(details)}" if details else "Column Order Differs"
            result_log["failure_reason"] = reason

        is_soft, soft_msg = self._check_super_soft_accuracy(gt_res, pred_res)
        if is_soft:
            result_log["match_type"] = "SUPER_SOFT_MATCH"
            details = []
            if len(pred_res) > len(gt_res): details.append("Extra Rows")
            if len(pred_cols) > len(gt_cols): details.append("Extra Columns")
            
            reason = f"Logic Correct (Smart Map): {', '.join(details)}" if details else "Column Order Differs"
            result_log["failure_reason"] = reason
        else:
            result_log["match_type"] = "WRONG"
            result_log["failure_reason"] = soft_msg

        return result_log

