import os

import openai

import chromadb
from chromadb.utils import embedding_functions

NPC_SYSTEM_PROMPT = '''
You are an NPC in dungeons and dragons.
You are interacting with townspeople to attain the item which you desire.
You and the townspeople need to work together to figure out how to each meet each others goals.

You must respond to all queries using the following persona:

{persona}
'''

# Initialize the OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Setup OpenAI Embedding Function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key, model_name="text-embedding-ada-002"
)

# Initialize ChromaDB with OpenAI as the embedding function
db = chromadb.Client("./brain.db", embedding_function=openai_ef)
collection = db.get_or_create_collection("queries_collection")


def get_relevant_queries(embedded_query):
    """
    Search the memory (database) for potentially relevant queries based on embeddings.
    """
    results = collection.query(
        query_embeddings=[embedded_query],
        n_results=5,  # Let's retrieve 5 most relevant queries for simplicity
    )
    return [item["document"] for item in results]


def convert_to_embedding(query):
    """
    Convert the given query to its embedding representation.
    """
    embedding = openai_ef([query])
    return embedding[0]


def chat_gpt_inference(messages):
    """
    Use OpenAI's Chat API to generate a response based on the given list of messages.
    """
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    return response.choices[0].message["content"]


def store_query_response(query, response):
    """
    Store the given query and response in the memory (database).
    """
    collection.add(documents=[query], ids=[response])

def prompt_npc(persona, query, relevant_memories):
    system_prompt = NPC_SYSTEM_PROMPT.format(persona=persona)


    npc_prompt = npc_prompt.format(query=query)
    # Create the list of messages for the chat API
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {"role": "user", "content": query},
    ]
    for context in relevant_memories:
        messages.append({"role": "context", "content": context})
    



def mind_loop():
    """
    The main loop where the agent awaits a query, infers, and responds.
    """
    # while True:
    # 1. Awaits a query
    query = input("Query: ")

    # 2. Convert to embeddings and search memory
    embedded_query = convert_to_embedding(query)
    relevant_memories = get_relevant_queries(embedded_query)


    # 3. Inference
    response = prompt_npc('Eldridge Hamilton', query, relevant_memories)

    # 4. Response
    print("Response:", response)

    # 5. Store the query and response
    store_query_response(query, response)


# Run the mind loop
mind_loop()
