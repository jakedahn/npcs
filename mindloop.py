import json
import hashlib
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
db = chromadb.PersistentClient(path="./brain.db")
collection = db.get_or_create_collection(name="npc_eldridge_hamilton", embedding_function=openai_ef)

def get_relevant_queries(embedded_query):
    """
    Search the memory (database) for potentially relevant queries based on embeddings.
    """
    results = collection.query(
        query_embeddings=[embedded_query],
        n_results=10,  # Let's retrieve 10 most relevant queries for simplicity
    )

    return results['documents']


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

    document = {'user': query, 'assistant': response}

    doc_id = hashlib.sha256(query.encode() + response.encode()).hexdigest()

    collection.add(documents=[json.dumps(document)], ids=[doc_id])

def get_townspeople_info(role):
    """
    Get the townspeople information from the memory (database).
    """
    # load ./npcs.json
    npcs = json.load(open('./npcs.json', 'r'))

    keys = [person['townspeople_keys'] for person in npcs if person['role'] == role]


    townspeople = []
    for npc in npcs:
        if npc['role'] == role:
            continue


        npc_info = {}
        for key in keys:
            npc_info[key] = npc[key]
        
        townspeople.append(npc_info)
    import ipdb; ipdb.set_trace()  # fmt: skip

# get_townspeople_info('ur mom')
get_townspeople_info('Town Mayor')

def prompt_npc(persona, query, relevant_memories):
    system_prompt = NPC_SYSTEM_PROMPT.format(persona='''{
    "role": "Town Mayor",
    "race": "Human",
    "name": "Eldridge Hamilton",
    "alignment": "Lawful Good",
    "world_scenario": "The town is facing a crisis as a bandit gang threatens to attack, and Eldridge is trying to organize defenses.",
    "description": "Eldridge is a wise, elder statesman, beloved by his people. He has a strong sense of justice and responsibility, having dedicated his life to his town's welfare.",
    "personality": "Eldridge is stoic, diplomatic, and steadfast. He is always ready to listen to the problems of his townsfolk and is deeply committed to finding solutions.",
    "inventory": ["Quality Ingot"],
    "desired_item": "Barrel of Ale",
    "will_sell_item": true,
    "first_mes": "Greetings, travelers. What brings you to our town in these trying times?",
    "mes_example": "We are doing everything we can to protect our people. I fear we may need assistance..."
}
''')

    # Create the list of messages for the chat API
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {"role": "user", "content": query},
    ]
    for ctx in relevant_memories:
        if not ctx:
            continue

        # Chroma, why you do this to me.
        memory = json.loads(ctx[0])

        messages.append({"role": 'user', "content": memory['user']})
        messages.append({"role": 'assistant', "content": memory['assistant']})

    return chat_gpt_inference(messages)


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
