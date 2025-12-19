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

# 3. Database Overlap (Unique DB Counts)
db_counts = {
    'Dataset': ['Train', 'Test (Dev)'],
    'Unique Databases': [train_df['db_id'].nunique(), test_df['db_id'].nunique()]
}
db_df = pd.DataFrame(db_counts)
print("\n--- Database Statistics ---")
print(db_df)

# 4. Evidence Usage (External Knowledge)
# BIRD dataset often uses 'evidence' column for extra context
train_has_evidence = train_df['evidence'].apply(lambda x: len(str(x)) > 0).sum()
test_has_evidence = test_df['evidence'].apply(lambda x: len(str(x)) > 0).sum()

print(f"\nQueries with External Knowledge (Evidence):")
print(f"Train: {train_has_evidence} ({train_has_evidence/len(train_df)*100:.1f}%)")
print(f"Test: {test_has_evidence} ({test_has_evidence/len(test_df)*100:.1f}%)")