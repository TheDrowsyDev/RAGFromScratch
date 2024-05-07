import argparse
from uuid import uuid4
import json
import requests
import os

tei_url = "http://localhost:8000"
chunk_size = 200

def main(text_file: str):
    data = None
    try:
        data = None
        with open(text_file, "r") as f:
            data = f.read()
    except Exception as e:
        print(f"Error: Unable to read file: {e}")
        exit(1)

    # Split data into chunks
    doc_chunks = []
    for idx in range(0, len(data), chunk_size):
        doc_chunks.append(data[idx:idx+chunk_size])

    # Filter empty strings
    doc_chunks = [doc for doc in doc_chunks if len(doc) != 0]

    # Update JSON file
    # First, read the original data
    orig_data = []

    if os.path.exists("db.json"):
        try:
            with open("db.json", "r") as f:
                orig_data = json.load(f)
        except Exception:
            print("Unable to read JSON data, going to override file...")

    for chunk in doc_chunks:
        # Get vector embedding
        data = {"inputs": [chunk]}
        response = requests.post(f"{tei_url}/embed", json=data)
        vector = response.json()[0]
        chunk_id = uuid4()
        orig_data.append({
            "chunk_id": f"{chunk_id}",
            "content": chunk,
            "embedding": vector
        })
    with open("db.json", "w") as f:
        json.dump(orig_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ingest")
    parser.add_argument("text_file", type=str, help="Text file to ingest.")

    args = parser.parse_args()
    text_file = args.text_file

    main(text_file)
