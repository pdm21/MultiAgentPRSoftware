from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from search_agent import SearchAgent, AgentState  # Importing AgentState from search_agent
from source_analysis_agent import SummarySentimentAgent  # Import the second agent

# Define system prompt for SearchAgent
search_prompt = """You are a smart and efficient research assistant working for a PR firm. 
Your task is to use the search engine to look up information about individuals, specifically by searching their name along with relevant keywords or topics. 
You are required to fetch the top 5 results that best match the user's query.  
If necessary, you may refine your search query to ensure accuracy, but avoid unnecessary searches. 
Present the results in a concise list format, prioritizing relevance and credibility.
Only gather text-based sources. Videos, images, gifs, and other such media should not be considered."""

# Instantiate Web-Search Agent
model = ChatOpenAI(model="gpt-4o")
search_params = {
    'filter': {
        'date': 'last_30_days',  
        'type': 'article'
    }
}
tool = TavilySearchResults(max_results=5, search_params=search_params)
search_agent = SearchAgent(model, [tool], system=search_prompt)
query = "Public perception of Angel Reese recent performances"
messages = [HumanMessage(content=query)]

# Step 1: Use SearchAgent to get the links
# Initialize the state with AgentState using the messages
initial_state = AgentState(messages=messages)
result = search_agent.graph.invoke(initial_state)
print(result['messages'][-1].content)