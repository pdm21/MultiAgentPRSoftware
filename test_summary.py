from dotenv import load_dotenv
_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Define the tools
tool = TavilySearchResults(max_results=5)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    urls: list[str]
    current_index: int
    chosen_articles: list[str]

class SummaryAgent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        
        # Define graph nodes
        graph.add_node("fetch_content", self.fetch_content)
        
        # Set an entry point for the graph
        graph.set_entry_point("fetch_content")

        # Compile graph and provide associated tools
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def fetch_content(self, state):
        """Fetches content and sources using Tavily's get_search_context."""
        query = state["urls"][state["current_index"]]
        print(f"Fetching content from URL: {query}")

        # Use get_search_context to retrieve context-rich content
        try:
            content = self.tools['tavily_search_results_json'].get_search_context(
                query=query,
                max_tokens=4000,  # Adjust token limit as needed
                search_depth=3,  # Look deeper into the website
                days=30,  # Focus on content from the past 30 days
                max_results=5,  # Limit to 5 sources
            )
            state["content"] = content
            print(f"Content successfully fetched ({len(content)} characters):")
            print(content[:500])  # Print first 500 characters for verification
        except Exception as e:
            print(f"Error fetching content: {e}")
            state["content"] = ""

        return state

    # def fetch_content(self, state):
    #     """Fetches the content from the current URL and saves it to the state."""
    #     url = state["urls"][state["current_index"]]
    #     print(f"Fetching content from URL: {url}")  

        
    #     response = self.tools['tavily_search_results_json'].invoke({"query": url})

        
    #     if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
    #         state["content"] = response[0].get("content", "")
    #         print("Content successfully fetched:")
    #         print(state["content"][:500])  
    #         print("Number of characters in content:", len(state["content"]))  

    #     else:
    #         print("No content found or unexpected response structure.")
    #         state["content"] = ""

    #     return state




# Set up the model and tool for testing
model = ChatOpenAI(model="gpt-4o")

# Initialize the agent with the model and tool
summary_agent = SummaryAgent(model, [tool])

# Test URLs to be fetched
urls_to_test = [
    "https://rollingout.com/2024/09/30/angel-reese-navigating-stardom-in-wnba/",
    "https://www.newsstation2.com/2024/11/14/just-in-wnba-athletes-have-valid-reasons-to-be-worried-about-the-recent-angel-reese-video/"
]

# Initialize the state with URLs and current index
state = {
    "urls": urls_to_test,
    "current_index": 0,
    "chosen_articles": []
}

# Test fetching content for each URL
for i in range(len(urls_to_test)):
    state["current_index"] = i 
    summary_agent.fetch_content(state)  
