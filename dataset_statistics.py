import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prep(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['set_label'] = label
    return df

# Load datasets
train_df = load_and_prep('data/train/train.json', 'Train')
test_df = load_and_prep('data/dev_20240627/dev.json', 'Test (Dev)')

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# 1. Difficulty Distribution (Test Data)
plt.figure(figsize=(12, 7))
sns.countplot(data=test_df, x='difficulty', 
              order=['simple', 'moderate', 'challenging'])

plt.title('Difficulty Distribution:Test Data')
plt.xlabel('Complexity Level')
plt.ylabel('Number of Queries')

ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

plt.savefig('comparison_difficulty.png')
plt.show()

# 2. Question Length Distribution
train_df['q_len'] = train_df['question'].apply(lambda x: len(x.split()))
test_df['q_len'] = test_df['question'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 6))
sns.kdeplot(train_df['q_len'], label='Train', fill=True)
sns.kdeplot(test_df['q_len'], label='Test (Dev)', fill=True)
plt.title('Question Length Distribution (Word Count)')
plt.xlabel('Number of Words')
plt.ylabel('Density')
plt.legend()
plt.savefig('comparison_length.png')
plt.show()

# 3. Database Overlap (Intersection Analysis)
train_dbs = set(train_df['db_id'].unique())
test_dbs = set(test_df['db_id'].unique())

common_dbs = train_dbs.intersection(test_dbs)
only_train = train_dbs - test_dbs
only_test = test_dbs - train_dbs

print(f"Databases in both sets: {len(common_dbs)}")
print(f"Databases ONLY in Train: {len(only_train)}")
print(f"Databases ONLY in Test (Unseen): {len(only_test)}")

# Visualization: Set Comparison
overlap_stats = pd.DataFrame({
    'Category': ['Common', 'Only Train', 'Only Test (Unseen)'],
    'Count': [len(common_dbs), len(only_train), len(only_test)]
})

plt.figure(figsize=(10, 6))
sns.barplot(data=overlap_stats, x='Category', y='Count', palette='coolwarm')
plt.title('Database Distribution: Seen vs Unseen')
plt.ylabel('Number of Databases')

for i, v in enumerate(overlap_stats['Count']):
    plt.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

plt.savefig('db_intersection_analysis.png')
plt.show()

# 4. Evidence Usage (External Knowledge)
# BIRD dataset often uses 'evidence' column for extra context
train_has_evidence = train_df['evidence'].apply(lambda x: len(str(x)) > 0).sum()
test_has_evidence = test_df['evidence'].apply(lambda x: len(str(x)) > 0).sum()

evidence_stats = pd.DataFrame({
    'Dataset': ['Train', 'Test (Dev)'],
    'Has Evidence': [train_has_evidence, test_has_evidence],
    'No Evidence': [len(train_df) - train_has_evidence, len(test_df) - test_has_evidence]
})
evidence_stats.set_index('Dataset').plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightgrey'])
plt.title('Proportion of Queries with External Knowledge (Evidence)')
plt.ylabel('Number of Queries')
plt.xticks(rotation=0)
plt.savefig('evidence_counts.png')
plt.show()

# 5. Correlation between Question and SQL length
train_df['sql_len'] = train_df['SQL'].apply(lambda x: len(x.split()))
correlation = train_df['q_len'].corr(train_df['sql_len'])

print(f"Correlation between Question and SQL length: {correlation:.2f}")

plt.figure(figsize=(10, 6))
sns.regplot(data=train_df.sample(500), x='q_len', y='sql_len', scatter_kws={'alpha':0.3})
plt.title('Question Length vs SQL Length (Sample of 500)')
plt.xlabel('Question Words')
plt.ylabel('SQL Words')
plt.savefig('correlation_between_sql_queery_length.png')
plt.show()

# 6. Query difficulty via keywords
keywords = ['JOIN', 'HAVING', 'UNION', 'GROUP BY', 'ORDER BY', 'WHERE', 'LIMIT', 'INTERSECT', 'EXCEPT']

def get_keyword_stats(df, label):
    total = len(df)
    stats = []
    for kw in keywords:
        count = df['SQL'].str.upper().str.contains(kw).sum()
        ratio = (count / total) * 100
        stats.append({'Keyword': kw, 'Ratio (%)': round(ratio, 2), 'Dataset': label})
    return stats

train_stats = get_keyword_stats(train_df, 'Train')
test_stats = get_keyword_stats(test_df, 'Test (Dev)')

stats_df = pd.DataFrame(train_stats + test_stats)

plt.figure(figsize=(14, 7))
sns.barplot(data=stats_df, x='Keyword', y='Ratio (%)', hue='Dataset')

plt.title('SQL Complexity Analysis: Keyword Usage Ratio (%)', fontsize=15)
plt.ylabel('Percentage of Queries (%)', fontsize=12)
plt.xlabel('SQL Keywords', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

ax = plt.gca()
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'%{p.get_height():.1f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('sql_keyword_ratios.png')
plt.show()

pivot_df = stats_df.pivot(index='Keyword', columns='Dataset', values='Ratio (%)')

# 7. Query difficulty via join and group by
def analyze_complexity(df, label):
    # Counting occurrences of keywords in each SQL string
    df['join_count'] = df['SQL'].str.upper().str.count('JOIN')
    df['groupby_count'] = df['SQL'].str.upper().str.count('GROUP BY')
    df['total_complexity'] = df['join_count'] + df['groupby_count']

    max_join = df['join_count'].max()
    max_gb = df['groupby_count'].max()
    
    # Finding the most complex SQL example
    most_complex_sql = df.loc[df['total_complexity'].idxmax(), 'SQL']
    most_complex_q = df.loc[df['total_complexity'].idxmax(), 'question']

    return {
        'Dataset': label,
        'Max JOINs': max_join,
        'Max GROUP BYs': max_gb,
        'Max Combined (JOIN+GB)': df['total_complexity'].max(),
        'Example Question': most_complex_q,
        'Example SQL': most_complex_sql
    }

train_results = analyze_complexity(train_df, 'Train')
test_results = analyze_complexity(test_df, 'Test (Dev)')

results_df = pd.DataFrame([train_results, test_results])

melted_results = results_df.melt(id_vars='Dataset', value_vars=['Max JOINs', 'Max GROUP BYs'])
plt.figure(figsize=(10, 6))
sns.barplot(data=melted_results, x='value', y='variable', hue='Dataset')
plt.title('Maximum Complexity Indicators')
plt.xlabel('Count')
plt.ylabel('')
plt.savefig('sql_maximum_counts.png')
plt.show()

# 8. Database Schema Density (Tables per DB)
with open('data/train/train_tables.json', 'r') as f:
    tables_data = json.load(f)

table_counts_df = pd.DataFrame([
    {'db_id': db['db_id'], 'table_count': len(db['table_names_original'])} 
    for db in tables_data
])

plt.figure(figsize=(10, 6))
# Boxplot shows median, quartiles and outliers clearly
sns.boxplot(x=table_counts_df['table_count'], color='lightgreen')
sns.stripplot(x=table_counts_df['table_count'], color='black', alpha=0.3) # Add dots for each DB

plt.title('How many tables are in a Database?')
plt.xlabel('Number of Tables')
plt.savefig('tables_per_db_boxplot.png')
plt.show()

# 9. Distribution of SELECT types (COUNT, SUM, AVG, or raw Columns)
def get_select_type(sql):
    sql = sql.upper()
    if 'SELECT COUNT' in sql: return 'COUNT'
    if 'SELECT SUM' in sql: return 'SUM'
    if 'SELECT AVG' in sql: return 'AVG'
    if 'SELECT MIN' in sql or 'SELECT MAX' in sql: return 'MIN/MAX'
    return 'COLUMN/VALUE'
train_df['select_type'] = train_df['SQL'].apply(get_select_type)
plt.figure(figsize=(8, 8))
train_df['select_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Distribution of SQL Operation Types (Train Set)')
plt.ylabel('') 
plt.savefig('distribution_select_types.png')
plt.show()