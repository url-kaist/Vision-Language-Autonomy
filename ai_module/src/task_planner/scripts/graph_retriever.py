import time
import re
import json

from ai_module.src.visual_grounding.scripts.vlms.loaders.client import LlmClient

class G_Retriever():
    def __init__(self, reference_graph):
        self.reference_graph = reference_graph
        self.reference_graph_json = self.reference_graph.to_json()

        # Load config from config.json
        with open("/ws/external/ai_module/src/config.json", "r") as f:
            config = json.load(f)
        
        self.client = LlmClient(model_name=config['MODEL_NAME'], api_key=config['GOOGLE_API_KEY0'])

    def retrieve(self, action: str, target_node_id: int, query_graph):
        # measure time
        start_time = time.time()

        query_graph_json = query_graph.to_json()

        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(action, target_node_id, query_graph_json, self.reference_graph_json)

        # make messages
        messages = [
            {
                "role": "system",
                "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt}
        ]

        # query LM
        response, _, usage_log = self.client.get_response(messages, 0)
        print(f"Response: {response}")

        matches = re.findall(r'Answer:\s*(\d+)', response)
        answer = matches[-1] if matches else None
        print(f"Answer: {answer}")

        # measure time
        processing_time = time.time() - start_time
        print(f"Processing time: {processing_time:.2f} sec")
        return answer

    def get_system_prompt(self):
        prompt = """
        Answer the following question based on the provided reference and query graphs. 
        Analyze the semantic and geometric relationships between nodes to generate your response.
        """
        return prompt

    def get_user_prompt(self, action, target_node_id, query_graph_json, reference_graph_json):
        # get question
        if action == 'count':
            question = f"Count how many objects in the reference graph match the entity of node ID {target_node_id} in the query graph, considering both semantic similarity and geometric relationships with other nodes in the query graph."
        elif action == 'find':
            question = f"Find the node ID in the reference graph that match node ID {target_node_id} in the query graph, considering both semantic similarity and geometric relationships with other nodes in the query graph."

        prompt = f"""
        question:
        {question}
        
        reference_graph:
        {reference_graph_json}
        
        query_graph:
        {query_graph_json}
        
        Please perform the following steps:
        1. Identify nodes in the reference_graph that best match the entities described in the query_graph.
        2. Reason about spatial or semantic relationships between these matched nodes.
        3. Think step: Analyze the semantic and geometric similarities, as well as the relationships between nodes, to determine the best match.
        4. Answer step: Provide a concise answer to the given question based on the analysis above.
            4.1. If the question is asking for a count, you must return only an integer number representing the count.
            4.2. If the question is asking to find objects or entities, you must return only a single 'node_id' from the matched nodes in the 'reference_graph' that is the most similar among the matched nodes, or return nothing if no match is found.
        
        You must format the response as follows:
        Think: [Provide reasoning and analysis based on the reference and query graphs.]
        Answer: [Provide the final answer in the required format (integer count or node_id).]
        """
        return prompt



