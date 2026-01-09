import json
import os
import argparse
from tqdm import tqdm
from onePassLlmModel.sql_compiler import JSONToSQLCompiler
from onePassLlmModel.sql_compiler import JSONToSQLCompiler
from bird_evaluator import BirdEvaluator

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
    parser.add_argument("--input_file", type=str, default="results/pipeline_test_report.jsonl", help="Girdi dosyası (.jsonl)")
    parser.add_argument("--output_file", type=str, default="results/pipeline_test_report_reprocessed_without_hint.jsonl", help="Çıktı dosyası (.jsonl)")
    parser.add_argument("--stats_file", type=str, default="results/pipeline_test_stats_reprocessed_without_hint.json", help="İstatistik dosyası (.json)")
    parser.add_argument("--db_path", type=str, default="financial.sqlite", help="Veritabanı dosyası yolu")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Hata: Girdi dosyası bulunamadı -> {args.input_file}")
        return

    # Evaluator'ı döngü dışında bir kez başlatıyoruz
    print(f"Evaluator başlatılıyor (DB: {args.db_path})...")
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
        "router_filtered": 0
    }

    print(f"İşleniyor: {args.input_file} -> {args.output_file}")

    with open(args.input_file, 'r', encoding='utf-8') as fin, \
         open(args.output_file, 'w', encoding='utf-8', buffering=1) as fout:

        lines = fin.readlines()

        for line in tqdm(lines, desc="Reprocessing"):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode hatası: {e}")
                continue

            # Temel verileri al
            query_id = str(data.get("question_id", "0")) 
            nl_query = data.get("query") or data.get("question")
            gt_sql = data.get("ground_truth_sql") or data.get("SQL")

            steps = data.get("steps", {})
            
            # 1. Router Elediyse Geç
            if data.get("status") == "filtered_by_router":
                update_stats(stats, data)
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # 2. Decomposer Verisi Yoksa Geç
            decomposer_res = steps.get("decomposer", {})
            json_plan = decomposer_res.get("json_plan")
            
            if not json_plan or decomposer_res.get("status") != "success":
                # Veri eksikse olduğu gibi yaz, istatistiğe ekle
                update_stats(stats, data)
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # --- A. COMPILER AŞAMASI (MANUEL ÇAĞRI) ---
            step_compiler = {"status": "pending", "generated_sql": None}
            generated_sql = None
            
            try:
                # Class instance oluştur ve compile et
                compiler = JSONToSQLCompiler(json_plan)
                generated_sql = compiler.compile()
                
                step_compiler["generated_sql"] = generated_sql
                
                # SQL comment ile başlıyorsa hata mesajıdır (senin kodundaki yapıya göre)
                if generated_sql and generated_sql.strip().startswith("--"):
                    step_compiler["status"] = "error_in_sql"
                    step_compiler["error"] = generated_sql # Hata mesajı SQL içinde dönüyor
                else:
                    step_compiler["status"] = "success"
                    
            except Exception as e:
                step_compiler["status"] = "error"
                step_compiler["error"] = str(e)

            steps["compiler"] = step_compiler

            # --- B. EVALUATOR AŞAMASI (MANUEL ÇAĞRI) ---
            step_evaluator = {"status": "skipped", "match_type": "NONE"}
            
            if step_compiler["status"] == "success" and generated_sql:
                try:
                    # Token stats varsa al, yoksa boş ver
                    token_stats = decomposer_res.get("token_stats", {})
                    
                    eval_res = evaluator.evaluate_query(
                        query_id=query_id,
                        natural_language_query=nl_query,
                        gt_sql=gt_sql,
                        pred_sql=generated_sql,
                        token_stats=token_stats
                    )
                    step_evaluator = eval_res
                    data["status"] = "completed"
                except Exception as e:
                    step_evaluator["status"] = "error"
                    step_evaluator["error"] = str(e)
                    data["status"] = "error"
            else:
                data["status"] = "error" # Compiler başaramadıysa genel durum error
            
            steps["evaluator"] = step_evaluator
            data["steps"] = steps
            
            # İstatistik güncelle ve kaydet
            update_stats(stats, data)

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            # fout.flush() # Buffering=1 olduğu için her satırda yazmayabilir ama güvenli olsun dersen açabilirsin

    # --- İSTATİSTİKLERİ DOSYAYA KAYDET ---
    try:
        os.makedirs(os.path.dirname(args.stats_file), exist_ok=True)
        with open(args.stats_file, 'w', encoding='utf-8') as f_stats:
            json.dump(stats, f_stats, indent=4)
        print(f"İstatistikler kaydedildi: {args.stats_file}")
    except Exception as e:
        print(f"İstatistik dosyası kaydedilemedi: {e}")

    print("\n" + "="*40)
    print("GÜNCELLEME TAMAMLANDI")
    print("="*40)
    print(json.dumps(stats, indent=4))
    print(f"\nYeni veri dosyası: {args.output_file}")

if __name__ == "__main__":
    main()