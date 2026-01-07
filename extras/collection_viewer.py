import chromadb

# 1. Connect to the existing database
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Pick a collection you want to see (e.g., "loan_status" or "district_A2")
collection_name = "district_A2" 

try:
    collection = client.get_collection(name=collection_name)
    
    # 3. Get the total count
    count = collection.count()
    print(f"Total items in '{collection_name}': {count}")

    # 4. Peek at the first 5 items (returns IDs, embeddings, metadatas, documents)
    # You can also use collection.get(limit=5) for more control
    data = collection.peek(limit=50)
    
    print("\n--- First 50 Items ---")
    for i in range(len(data['ids'])):
        print(f"ID: {data['ids'][i]}")
        print(f"Document (What is embedded): {data['documents'][i]}")
        print(f"Metadata (Original DB Data): {data['metadatas'][i]}")
        print("-" * 30)

except ValueError:
    print(f"Collection '{collection_name}' does not exist.")