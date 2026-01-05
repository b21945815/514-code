import json

def get_text_columns_financial(file_path, target_db_id='financial'):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Find the specific database
    target_db = next((item for item in data if item["db_id"] == target_db_id), None)

    if not target_db:
        print(f"Database '{target_db_id}' not found in the file.")
        return

    # Mappings for clearer output
    tables = target_db['table_names_original']
    col_names = target_db['column_names_original']
    col_types = target_db['column_types']

    # Dictionary to hold results: { "table_name": [list_of_columns] }
    text_columns_by_table = {table: [] for table in tables}

    # Iterate through all columns
    # The arrays column_names_original, column_names, and column_types are aligned by index.
    for i, col_type in enumerate(col_types):
        if col_type == 'text':
            # col_names[i] is a list like [table_index, "column_name"]
            table_idx = col_names[i][0]
            col_name = col_names[i][1]

            # Ignore the wildcard "*" column (usually index -1)
            if table_idx >= 0:
                table_name = tables[table_idx]
                text_columns_by_table[table_name].append(col_name)

    # Print results
    print(f"--- Text Columns in '{target_db_id}' Database ---")
    for table, columns in text_columns_by_table.items():
        if columns:
            print(f"\nTable: {table}")
            for col in columns:
                print(f"  - {col}")

if __name__ == "__main__":
    # Ensure this matches your actual file name
    file_name = 'data/dev_20240627/dev_tables.json'
    get_text_columns_financial(file_name)