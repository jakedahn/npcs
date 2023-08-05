import time
import json
import hashlib
import os

import openai

import chromadb
from chromadb.utils import embedding_functions
from termcolor import colored

NPC_SYSTEM_PROMPT = '''
You are an NPC in dungeons and dragons.
You are interacting with townspeople to attain the item which you desire.
You and the townspeople need to work together to figure out how to each meet each others goals.

You must respond to all queries, in first person, using the following persona:

{persona}

You know of the following townspeople:

{townspeople_context}
'''

NPCS = json.load(open('./npcs.json', 'r'))

class NPC:
    def __init__(self, name):
        self.name = name
        self.persona = self.get_persona()

        formatted_name = name.lower().replace(' ', '_')
        # Initialize the OpenAI API
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # Setup OpenAI Embedding Function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai.api_key, model_name="text-embedding-ada-002"
        )

        # Initialize ChromaDB with OpenAI as the embedding function
        self.db = chromadb.PersistentClient(path="./brain.db")
        self.collection = self.db.get_or_create_collection(name=f"npc_{formatted_name}", embedding_function=self.openai_ef)

    def get_relevant_queries(self, embedded_query):
        """
        Search the memory (database) for potentially relevant queries based on embeddings.
        """
        results = self.collection.query(
            query_embeddings=[embedded_query],
            n_results=10,  # Let's retrieve 10 most relevant queries for simplicity
        )

        return results['documents']


    def convert_to_embedding(self, query):
        """
        Convert the given query to its embedding representation.
        """
        embedding = self.openai_ef([query])
        return embedding[0]


    def chat_gpt_inference(self, messages):
        """
        Use OpenAI's Chat API to generate a response based on the given list of messages.
        """
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        return response.choices[0].message["content"]


    def store_query_response(self, query, response):
        """
        Store the given query and response in the memory (database).
        """

        document = {'user': query, 'assistant': response}

        doc_id = hashlib.sha256(query.encode() + response.encode() + str(time.time()).encode()).hexdigest()

        self.collection.add(documents=[json.dumps(document)], ids=[doc_id])

    def get_persona(self):
        return list(filter(lambda npc: npc['name'] == self.name, NPCS))[0]

    def get_townspeople_info(self):
        """
        Get the townspeople information from the memory (database).
        """

        my_lookup_keys = self.persona['townspeople_keys']
        townspeople = []

        for npc in NPCS:
            if npc['name'] == self.name:
                continue

            npc_info = {}
            for key in my_lookup_keys:
                npc_info[key] = npc[key]
            
            townspeople.append(npc_info)

        return townspeople

    def prompt_npc(self, query, relevant_memories):
        townspeople_context = self.get_townspeople_info()
        system_prompt = NPC_SYSTEM_PROMPT.format(persona=self.persona, townspeople_context=townspeople_context)

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

        return self.chat_gpt_inference(messages)


    def mind_loop(self, query):
        """
        The main loop where the agent awaits a query, infers, and responds.
        """
        # 2. Convert to embeddings and search memory
        embedded_query = self.convert_to_embedding(query)
        relevant_memories = self.get_relevant_queries(embedded_query)

        # 3. Inference
        response = self.prompt_npc(query, relevant_memories)

        # 4. Response
        print(colored(f'<{self.name}>', self.persona['color']))
        print(f'{response}\n\n')

        # 5. Store the query and response
        self.store_query_response(query, response)

        return response


eldridge = NPC('Eldridge Hamilton')
fizzbang = NPC('Fizzbang Whizzlegear')

query = 'hello moto'

for i in range(10):
    query = eldridge.mind_loop(query)
    query = fizzbang.mind_loop(query)
