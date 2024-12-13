from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool
from typing import Annotated, TypedDict, Dict
from perplexity_agent import PerplexityAPI
import operator
import os
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()


# Agent State definition
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# PerplexityTool definition
class PerplexityTool():
    def __init__(self, api_key: str):
        """
        Initializes the PerplexityTool with the provided API key.

        Args:
            api_key (str): The API key for Perplexity API.
        """
        self.api_client = PerplexityAPI(api_key)

    def invoke(self, args: Dict):
        """
        Fetches the most recent articles about a given query using PerplexityAPI.

        Args:
            args (Dict): A dictionary containing:
                - "query" (str): The search query or topic.
                - "num_articles" (int): The number of articles to fetch.

        Returns:
            list: A list of articles related to the query.
        """
        query = args.get("query")
        num_articles = args.get("num_articles", 5)  # Default to 5 articles if not provided
        return self.api_client.fetch_recent_articles(query, max_results=num_articles)


# AthleteArticleAgent class
class AthleteArticleAgent:
    def __init__(self, api_key: str):
        """
        Function: Initializes the AthleteArticleAgent.
        Args:     api_key (str): API key for Perplexity API.
        """
        # Define tools
        self.perplexity_tool = PerplexityTool(api_key)

        # Initialize the state graph
        graph = StateGraph(AgentState)

        # Add nodes and edges for the graph
        graph.add_node("llm", self.call_perplexity)
        graph.add_edge("llm", END)

        graph.set_entry_point("llm")


        # Compile the graph
        self.graph = graph.compile()

        """
        Function: Calls the PerplexityTool with the query from the state.
        Args:     state (AgentState): The current agent state.
        Returns:  AgentState: Updated state with tool response.
        """
    def call_perplexity(self, state: AgentState):
        
        last_message = state["messages"][-1] if state["messages"] else None

        # getting duplicates, trying to avoid this
        if isinstance(last_message, ToolMessage) and last_message.name == "PerplexityTool":
            return state

        if isinstance(last_message, HumanMessage):
            query = last_message.content
            articles = self.perplexity_tool.invoke({"query": query, "num_articles": 5})

            tool_call_id = str(uuid4())
            # Add tool response to the state
            state["messages"].append(
                ToolMessage(
                    name="PerplexityTool",
                    content="\n".join(articles),
                    tool_call_id=tool_call_id
                )
            )
        return state

    def run(self, query: str):
        """
        Executes the graph to fetch articles.

        Args:
            query (str): The search query.

        Returns:
            list: The list of articles fetched by the tool.
        """
        state = {"messages": [HumanMessage(content=query)]}
        final_state = self.graph.invoke(state)
        return final_state["messages"]


# Example Usage
if __name__ == "__main__":
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        raise ValueError("API key for Perplexity API is missing. Check your .env file.")

    agent = AthleteArticleAgent(api_key)
    result = agent.run("Lionel Messi")

    print("\nTool Messages:")
    for message in result:
        if isinstance(message, ToolMessage):
            print(message.content)
