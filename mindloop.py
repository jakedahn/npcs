import os
import openai
from chromadb import ChromaDB
from chromadb.utils import embedding_functions

# Initialize the OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Setup OpenAI Embedding Function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key, model_name="text-embedding-ada-002"
)

# Initialize ChromaDB with OpenAI as the embedding function
db = ChromaDB("./brain.db", embedding_function=openai_ef)


def get_relevant_queries(embedded_query):
    """
    Search the memory (database) for potentially relevant queries based on embeddings.
    """
    # Assuming ChromaDB has a method to search by embeddings
    relevant_queries = db.search_by_embedding(embedded_query)
    return relevant_queries


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
    db.insert({"query": query, "response": response})


def mind_loop():
    """
    The main loop where the agent awaits a query, infers and responds.
    """
    while True:
        # 1. Awaits a query
        query = input("Query: ")

        # 2. Convert to embeddings and search memory
        embedded_query = convert_to_embedding(query)
        relevant_contexts = get_relevant_queries(embedded_query)

        # Create the list of messages for the chat API
        messages = [
            {
                "role": "system",
                "content": "You are an NPC. Your sole purpose is to act as the character you are prompted to be. Respond in character to the query.",
            },
            {"role": "user", "content": query},
        ]
        for context in relevant_contexts:
            messages.append({"role": "context", "content": context})

        # 3. Inference
        response = chat_gpt_inference(messages)

        # 4. Response
        print("Response:", response)

        # 5. Store the query and response
        store_query_response(query, response)


# Run the mind loop
mind_loop()
