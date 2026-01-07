import json
import os
import argparse
from tqdm import tqdm
from onePassLlmModel.bird_pipeline import BirdSQLPipeline

def load_test_data(filepath):
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    financial_data = [d for d in data if d.get('db_id') == 'financial']
    
    return financial_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/dev_20240627/dev.json")
    parser.add_argument("--data_path_2", type=str, default="data/dev_20240627/dev_tied_append.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/pipeline_test_report.json")
    args = parser.parse_args()

    pipeline = BirdSQLPipeline()

    test_data = load_test_data(args.data_path)
    test_data_2 = load_test_data(args.data_path_2)
    test_data.extend(test_data_2)
    if not test_data:
        return
    
    if args.limit > 0:
        test_data = test_data[:args.limit]
    

    results = []
    stats = {
        "total": 0,
        "success": 0,
        "exact_match": 0,
        "strict_match": 0,
        "soft_match": 0,
        "super_soft_match": 0,
        "wrong": 0,
        "errors": 0,
        "router_filtered": 0
    }

    for item in tqdm(test_data, desc="Processing Queries"):
        stats["total"] += 1
        query = item['question']
        gt_sql = item['SQL']
        
        res = pipeline.process_query(query, ground_truth_sql=gt_sql)
        results.append(res)
        
        status = res.get("status")
        
        if status == "filtered_by_router":
            stats["router_filtered"] += 1
        
        elif status == "completed":
            match_type = res["steps"]["evaluator"].get("match_type")
            
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
            else:
                stats["wrong"] += 1
        else:
            stats["errors"] += 1

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": stats,
            "details": results
        }, f, indent=4)

if __name__ == "__main__":
    main()