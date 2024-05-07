import requests
import json
from openai import OpenAI
import numpy as np
from numpy.linalg import norm

rerank_url = "http://localhost:8002"
tgi_url = "http://localhost:8001"
tei_url = "http://localhost:8000"
threshold = 0.3 # Play around with higher or lower settings!
max_chunks = 15 # Max chunks for initial cosine search
rerank_chunks = 5 # Max chunks from the re-ranker
rag_prompt = """
### CONTEXT
{context}

Given the above context, answer the user's query below.
Query: {query}
Answer:
"""

llm = OpenAI(
    api_key="EMPTY",
    base_url=f"{tgi_url}/v1"
)


def main(rag_data: list[dict]):
    while True:
        query = input("Enter a query > ")
        if query == "exit":
            exit(0)
        
        # Vectorize user's original query
        data = {"inputs": [query]}
        response = requests.post(f"{tei_url}/embed", json=data)
        vector = response.json()[0]

        # Calculate cosine similarity to RAG documents
        for elem in rag_data:
            chunk_content = elem.get("embedding")

            cosine_sim = (np.dot(vector, chunk_content))/(norm(vector)*norm(chunk_content))
            elem['cosine_sim'] = cosine_sim
        
        # Set a cutoff for similarity
        filtered_context_chunks = [elem['content'] for elem in rag_data if elem['cosine_sim'] > threshold]
        if len(filtered_context_chunks) > max_chunks:
            filtered_context_chunks = filtered_context_chunks[:max_chunks]

        # Rerank the contexts by most relevant
        data = {
            "query": query,
            "texts": filtered_context_chunks
        }
        response = requests.post(f"{rerank_url}/rerank", json=data)
        ranks = response.json()

        # Grab the top three most relevant
        if len(ranks) > rerank_chunks:
            ranks = ranks[:rerank_chunks]

        indices = [elem['index'] for elem in ranks]
        selected_chunks = []
        for idx in indices:
            selected_chunks.append(filtered_context_chunks[idx])
        context = "\n".join(selected_chunks)

        response = llm.chat.completions.create(
            model="mistral",
            messages=[
                {
                    "role": "user",
                    "content": rag_prompt.format(
                        context=context,
                        query=query
                    )
                }
            ],
            stream=True,
            max_tokens=1024,
            temperature=0.1
        )

        for elem in response:
            content = elem.choices[0].delta.content
            print(content, end="")
        print("\n")
    

if __name__ == "__main__":
    rag_data = None
    with open("db.json", "r") as f:
        rag_data = json.load(f)
    main(rag_data)
