import chromadb

# Create or connect to a persistent ChromaDB client
client = chromadb.PersistentClient(path="path_to_your_chroma_db")

# List all existing collections
collections = client.list_collections()

# Print collection names
for collection in collections:
    print("Collection:", collection.name)
