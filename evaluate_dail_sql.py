import json
import os
import argparse
import pandas as pd
from tqdm import tqdm
from bird_evaluator import BirdEvaluator

def load_ground_truth(dev_path, append_path):
    if not os.path.exists(dev_path):
        print(f"Warning: {dev_path} not found.")
        return pd.DataFrame()

    df_main = pd.read_json(dev_path)
    
    if os.path.exists(append_path):
        df_append = pd.read_json(append_path)
        full_df = pd.concat([df_main, df_append], ignore_index=True)
    else:
        full_df = df_main

    financial_df = full_df[full_df['db_id'] == 'financial'].reset_index(drop=True)
    
    if 'question_id' in financial_df.columns:
        financial_df['question_id'] = financial_df['question_id'].astype(str)
    else:
        financial_df['question_id'] = financial_df.index.astype(str)

    return financial_df

def clean_dail_sql(sql_str):
    if not isinstance(sql_str, str):
        return ""
    
    if "\t----- bird -----" in sql_str:
        sql_str = sql_str.split("\t----- bird -----")[0]
    
    sql_str = sql_str.replace('\n', ' ').strip().rstrip(';')
    return sql_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default="baselines/dail_sql/RESULTS_MODEL-gpt-4o.json")
    parser.add_argument("--dev_path", type=str, default="data/dev_20240627/dev.json")
    parser.add_argument("--append_path", type=str, default="data/dev_20240627/dev_tied_append.json")
    parser.add_argument("--db_path", type=str, default="financial.sqlite")
    parser.add_argument("--output", type=str, default="results/dail_sql_final_report.json")
    args = parser.parse_args()

    print("Loading ground truth data...")
    gt_df = load_ground_truth(args.dev_path, args.append_path)
    if gt_df.empty:
        print("Error: No ground truth data found!")
        return

    print(f"Loading predictions from {args.pred_file}...")
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # Evaluator Başlat
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
        "missing": 0
    }
    
    results_detail = []

    print("Starting evaluation...")
    
    for _, row in tqdm(gt_df.iterrows(), total=len(gt_df)):
        q_id = str(row.get('question_id', ''))
        nl_query = row.get('question', '')
        gt_sql = row['SQL']
        
        # DAIL-SQL çıktısından tahmini çek
        pred_sql_raw = predictions.get(q_id)
        
        if pred_sql_raw is None:
            stats["total"] += 1
            stats["missing"] += 1
            results_detail.append({
                "question_id": q_id,
                "status": "missing_prediction"
            })
            continue

        pred_sql = clean_dail_sql(pred_sql_raw)
        
        # --- DOĞRU PARAMETRELERLE EVALUATE ÇAĞRISI ---
        eval_result = evaluator.evaluate_query(
            query_id=q_id,
            natural_language_query=nl_query,
            gt_sql=gt_sql,
            pred_sql=pred_sql,
            token_stats={}  # DAIL-SQL token sayısını bu JSON'da tutmuyor, boş geçiyoruz
        )
        
        stats["total"] += 1
        match_type = eval_result.get("match_type", "WRONG")
        execution_status = eval_result.get("execution_status", "UNKNOWN")
        
        if execution_status == "SQL_ERROR":
            stats["errors"] += 1
        elif match_type in ["EXACT_MATCH", "STRICT_EXACT_MATCH", "SOFT_MATCH", "SUPER_SOFT_MATCH"]:
            stats["success"] += 1
            if match_type == "EXACT_MATCH": stats["exact_match"] += 1
            elif match_type == "STRICT_EXACT_MATCH": stats["strict_match"] += 1
            elif match_type == "SOFT_MATCH": stats["soft_match"] += 1
            elif match_type == "SUPER_SOFT_MATCH": stats["super_soft_match"] += 1
        else:
            stats["wrong"] += 1
            
        results_detail.append(eval_result)

    # İstatistik Hesaplama
    if stats["total"] > 0:
        stats["accuracy"] = (stats["success"] / stats["total"]) * 100
    else:
        stats["accuracy"] = 0

    # Kaydet
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": stats,
            "details": results_detail
        }, f, indent=4, ensure_ascii=False)

    print("\n" + "="*40)
    print(f"EVALUATION COMPLETE")
    print("="*40)
    print(f"Total Queries: {stats['total']}")
    print(f"Success:       {stats['success']} ({stats['accuracy']:.2f}%)")
    print(f"  - Exact:     {stats['exact_match']}")
    print(f"  - Strict:    {stats['strict_match']}")
    print(f"  - Soft:      {stats['soft_match']}")
    print(f"  - Super Soft:{stats['super_soft_match']}")
    print(f"Errors:        {stats['errors']}")
    print(f"Missing Preds: {stats['missing']}")
    print("="*40)
    print(f"Detailed report saved to: {args.output}")

if __name__ == "__main__":
    main()