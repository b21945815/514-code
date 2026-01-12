import json
import os
import sys
sys.path.append(os.getcwd())
import argparse
import pandas as pd
from tqdm import tqdm
from bird_evaluator import BirdEvaluator

def load_ground_truth(dev_path, append_path):
    if not os.path.exists(dev_path) or not os.path.exists(append_path):
        return []


    df_main = pd.read_json(dev_path)
    df_append = pd.read_json(append_path)
    full_df = pd.concat([df_main, df_append], ignore_index=True)
    financial_df = full_df[full_df['db_id'] == 'financial'].reset_index(drop=True)
    
    return financial_df

def parse_prediction_string(pred_str):
    if not pred_str:
        return "SELECT * FROM table" 
    
    separator = "\t----- bird -----\t"
    if separator in pred_str:
        pred_str = pred_str.split(separator)[0].strip()
    separator = ";"
    if separator in pred_str:
        pred_str = pred_str.split(separator)[0].strip()
    return pred_str.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="baselines/din_sql/results.json")
    parser.add_argument("--dev_path", type=str, default="data/dev_20240627/dev.json")
    parser.add_argument("--append_path", type=str, default="data/dev_20240627/dev_tied_append.json")
    parser.add_argument("--db_path", type=str, default="data/dev_20240627/dev_databases/financial/financial.sqlite")
    parser.add_argument("--output", type=str, default="results/din_sql_final_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        print(f"Prediction file can not be found: {args.pred_file}")
        return

    with open(args.pred_file, 'r', encoding='utf-8') as f:
        predictions_dict = json.load(f)

    gt_df = load_ground_truth(args.dev_path, args.append_path)
    print(f"Total question: {len(gt_df)}")
    print(f"Model answered: {len(predictions_dict)}")

    evaluator = BirdEvaluator(db_filename=args.db_path)

    stats = {
        "total": 0,
        "success": 0,
        "exact_match": 0,
        "strict_match": 0,
        "soft_match": 0,
        "super_soft_match": 0,
        "wrong": 0,
        "errors": 0,
        "missing_prediction": 0
    }
    
    results = []

    for idx, row in tqdm(gt_df.iterrows(), total=len(gt_df), desc="Evaluating DIN-SQL"):
        stats["total"] += 1
        
        question_id = row.get('question_id', idx)
        query = row['question']
        gt_sql = row['SQL']
        
        str_idx = str(idx)
        
        if str_idx not in predictions_dict:
            stats["missing_prediction"] += 1
            results.append({
                "question_id": question_id,
                "status": "MISSING_PREDICTION",
                "gt_sql": gt_sql
            })
            continue

        raw_pred = predictions_dict[str_idx]
        pred_sql = parse_prediction_string(raw_pred)

        eval_res = evaluator.evaluate_query(
            query_id=question_id,
            natural_language_query=query,
            gt_sql=gt_sql,
            pred_sql=pred_sql
        )
        
        match_type = eval_res.get("match_type")
        
        if match_type == "EXACT_MATCH":
            stats["exact_match"] += 1
            stats["success"] += 1
        elif match_type == "STRICT_EXACT_MATCH":
            stats["strict_match"] += 1
            stats["success"] += 1
        elif match_type == "SOFT_MATCH":
            stats["soft_match"] += 1
            stats["success"] += 1
        elif match_type == "SUPER_SOFT_MATCH":
            stats["super_soft_match"] += 1
            stats["success"] += 1
        elif match_type == "SQL_ERROR":
            stats["errors"] += 1
        else:
            stats["wrong"] += 1

        results.append({
            "index": idx,
            "question": query,
            "gt_sql": gt_sql,
            "pred_sql": pred_sql, 
            "raw_pred": raw_pred, 
            "eval_result": eval_res
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": stats,
            "details": results
        }, f, indent=4)


    print("\n" + "="*40)
    print("🧪 DIN-SQL BASELINE")
    print("="*40)
    print(f"Toplam: {stats['total']}")
    print(f"Success:   {stats['success']}")
    print(f"Wrong:     {stats['wrong']}")
    print(f"Error:       {stats['errors']}")
    print(f"Missing:      {stats['missing_prediction']}")
    
if __name__ == "__main__":
    main()