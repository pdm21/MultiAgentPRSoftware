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
        """Fetches the content from the current URL and saves it to the state."""
        url = state["urls"][state["current_index"]]
        print(f"Fetching content from URL: {url}")  # Debug print statement
        
        # Invoke the tool to get the content
        response = self.tools['tavily_search_results_json'].invoke({"query": url})

        # Check if response is a list and contains dictionaries with 'content'
        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
            state["content"] = response[0].get("content", "")
            print("Content successfully fetched:")
            print(state["content"][:5000])  # Print the first 500 characters for verification
            print("Number of characters in content:", len(state["content"]))  # Print character count

        else:
            print("No content found or unexpected response structure.")
            state["content"] = ""

        return state




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
