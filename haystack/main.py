import os
from typing import Any, Dict, List, Tuple

from haystack import BaseComponent
from haystack.agents import Agent, Tool
from haystack.agents.base import ToolsManager
from haystack.nodes import PromptNode


class AdderNode(BaseComponent):
    outgoing_edges = 1
    
    def run(self, query: Dict[str, Any] = None):
        """
        Run the adder node with input from query dictionary
        
        :param query: Dictionary containing 'num1' and 'num2'
        :return: Tuple of (output dict, output edge name)
        """
        print(f'Received query in adderNode: {query}')
        if not query or 'num1' not in query or 'num2' not in query:
            raise ValueError("Query must contain 'num1' and 'num2' parameters")
            
        num1 = float(query['num1'])
        num2 = float(query['num2'])
        return {"sum": num1 + num2}, "output_1"

    def run_batch(
        self,
        num1: List[float],
        num2: List[float],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Add pairs of numbers in batch mode.
        
        :param num1: List of first numbers
        :param num2: List of second numbers
        :return: Tuple of (output dict, output edge name)
        """
        results = [float(n1) + float(n2) for n1, n2 in zip(num1, num2)]
        output = {
            "sums": results,
            "inputs": {"num1": num1, "num2": num2}
        }
        return output, "output_1"

def main():
    # Initialize prompt node (using default OpenAI model)
    prompt_node = PromptNode(
        "gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    add_tool = Tool(
        name="add",
        description="Add two numbers",
        pipeline_or_node=AdderNode(),
    )
    calculator_agent = Agent(
        prompt_node=prompt_node,
        tools_manager=ToolsManager([add_tool]),
    )
    
    # Example queries
    queries = [
        "Add (2 + 5) and (3 + 4), then add the results",
    ]
    
    # Process queries
    for query in queries:
        print(f"\nQuery: {query}")
        response = calculator_agent.run(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
