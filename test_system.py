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

def update_stats(stats, res):
    stats["total"] += 1
    status = res.get("status")
    
    if status == "filtered_by_router":
        stats["router_filtered"] += 1
    
    elif status == "completed":
        if "steps" in res and "evaluator" in res["steps"]:
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
    else:
        stats["errors"] += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/dev_20240627/dev.json")
    parser.add_argument("--data_path_2", type=str, default="data/dev_20240627/dev_tied_append.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/pipeline_test_report_with_hint.jsonl")
    parser.add_argument("--hint", type=bool, default=False)
    args = parser.parse_args()

    pipeline = BirdSQLPipeline(model="groq")

    test_data = load_test_data(args.data_path)
    test_data_2 = load_test_data(args.data_path_2)
    test_data.extend(test_data_2)
    
    if not test_data:
        return
    
    if args.limit > 0:
        test_data = test_data[:args.limit]
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
    print(len(test_data))
    processed_count = 0
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        prev_res = json.loads(line)
                        update_stats(stats, prev_res)
                        processed_count += 1
                    except json.JSONDecodeError:
                        continue
    
    if processed_count < len(test_data):
        test_data_to_process = test_data[processed_count:]
    else:
        return
    # If you are writing to file that has data for the first line new line for the first new line will probably be missing
    with open(args.output, 'a', encoding='utf-8', buffering=1) as f:
        for item in tqdm(test_data_to_process, desc="Processing Queries"):            
            query = item['question']
            gt_sql = item['SQL']
            args.hint = True
            if args.hint:
                hint = item.get('evidence')
            else:
                hint = None
            res = pipeline.process_query(query, ground_truth_sql=gt_sql, hint=hint)

            update_stats(stats, res)

            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush() 


 
    print(json.dumps(stats, indent=4))
    
    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()