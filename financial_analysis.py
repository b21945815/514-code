import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# 1. LOAD DATA
# Ensure these paths match your local directory structure
with open('data/dev_20240627/dev.json', 'r', encoding='utf-8') as f:
    dev_queries = json.load(f)

with open('data/dev_20240627/dev_tables.json', 'r', encoding='utf-8') as f:
    tables_data = json.load(f)

# 2. FILTER FOR 'FINANCIAL' DATABASE
fin_queries = [q for q in dev_queries if q['db_id'] == 'financial']
fin_tables = [t for t in tables_data if t['db_id'] == 'financial'][0]

print(f"--- Financial Dataset Overview ---")
print(f"Total Questions: {len(fin_queries)}")
print(f"Total Tables: {len(fin_tables['table_names_original'])}")
print(f"Total Columns: {len(fin_tables['column_names_original'])}")
print(f"Total Foreign Keys: {len(fin_tables['foreign_keys'])}")

# 3. SCHEMA STATISTICS (Columns per Table)
table_names = fin_tables['table_names_original']
col_counts = {}
for table_idx, col_name in fin_tables['column_names_original']:
    if table_idx == -1: continue
    t_name = table_names[table_idx]
    col_counts[t_name] = col_counts.get(t_name, 0) + 1

col_df = pd.DataFrame(list(col_counts.items()), columns=['Table', 'Column Count'])

plt.figure(figsize=(12, 6))
sns.barplot(data=col_df.sort_values('Column Count', ascending=False), x='Table', y='Column Count')
plt.title('Column Count per Table (Financial DB)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('financial_tables_columns.png')
plt.show()

# 4. SQL COMPLEXITY ANALYSIS
fin_df = pd.DataFrame(fin_queries)
fin_df['join_count'] = fin_df['SQL'].str.upper().str.count('JOIN')

plt.figure(figsize=(8, 5))
sns.countplot(data=fin_df, x='join_count', palette='plasma')
plt.title('Query Complexity: Number of JOINs')
plt.xlabel('JOIN Count')
plt.ylabel('Number of Queries')
plt.savefig('financial_query_complexity.png')
plt.show()

# 5. DATA CARDINALITY ANALYSIS (Without ID columns)
db_path = 'data/dev_20240627/dev_databases/financial/financial.sqlite'

try:
    conn = sqlite3.connect(db_path)
    cardinality_results = []

    for table in table_names:
        # Get total row count
        row_count = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM `{table}`", conn)['cnt'][0]
        
        # Get columns for this table
        cursor = conn.execute(f"PRAGMA table_info(`{table}`)")
        columns = [col[1] for col in cursor.fetchall()]
        
        for col in columns:
            if "id" in col.lower():
                continue
            # Count distinct values
            unique_vals = pd.read_sql_query(f"SELECT COUNT(DISTINCT `{col}`) as u_cnt FROM `{table}`", conn)['u_cnt'][0]
            ratio = unique_vals / row_count if row_count > 0 else 0
            
            cardinality_results.append({
                'Table': table,
                'Column': col,
                'Unique_Values': unique_vals,
                'Rows': row_count,
                'Ratio': round(ratio, 4)
            })

    card_df = pd.DataFrame(cardinality_results)
    
    # Visualization: Top Cardinality Ratios
    plt.figure(figsize=(12, 6))
    top_card = card_df.sort_values('Ratio', ascending=False).head(15)
    sns.barplot(data=top_card, x='Ratio', y='Column', hue='Table', dodge=False)
    plt.title('Top 15 Columns by Data Cardinality (Ratio)')
    plt.xlabel('Cardinality Ratio (Unique/Total)')
    plt.tight_layout()
    plt.savefig('financial_cardinality.png')
    plt.show()

    conn.close()
    print("\n--- Cardinality Analysis Completed ---")
    print(card_df.sort_values('Ratio', ascending=False).head(10).to_string(index=False))

except Exception as e:
    print(f"\n[Error connecting to SQLite]: {e}")
    print("Ensure the .sqlite file is in the correct directory.")

# 6. FOREIGN KEY MAPPING
print("\n--- Foreign Key Relationships ---")
for fk in fin_tables['foreign_keys']:
    # Mapping indices back to readable names
    col_idx, ref_col_idx = fk
    col_name = fin_tables['column_names_original'][col_idx][1]
    ref_col_name = fin_tables['column_names_original'][ref_col_idx][1]
    
    # Finding table names for these columns
    table_idx = fin_tables['column_names_original'][col_idx][0]
    ref_table_idx = fin_tables['column_names_original'][ref_col_idx][0]
    
    print(f"Connection: {table_names[table_idx]}.{col_name} -> {table_names[ref_table_idx]}.{ref_col_name}")