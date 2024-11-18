from dotenv import load_dotenv                                                                                                      # type: ignore
_ = load_dotenv()
from langgraph.graph import StateGraph, END                                                                                         # type: ignore
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage                                # type: ignore
from langchain_community.chat_models import ChatOpenAI                                                                  # type: ignore
from langchain_community.tools.tavily_search import TavilySearchResults # type: ignore
from langchain_openai import ChatOpenAI # type: ignore

"""
# Define the tools
- Do I need an external summary tool? Custom tool? Or no tool?
"""
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
        
        # define graph nodes
        graph.add_node("fetch_content", self.fetch_content)
        # graph.add_node("summarize", self.summarize_content)
        # graph.add_node("human_review", self.human_review)
        
        # define graph edges
        # graph.add_conditional_edges("fetch_content", self.has_more_urls, {True: "summarize", False: END})
        # graph.add_edge("summarize", "human_review")
        # graph.add_edge("human_review", "fetch_content")

        # set an entry point for the graph        
        graph.set_entry_point("fetch_content")

        # compile graph and provide associated tools
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def fetch_content(self, state):
        """Fetches the content from the current URL and saves it to the state."""
        url = state["urls"][state["current_index"]]
        print(f"Fetching content from URL: {url}")  # Debug print statement

        # Invoke the tool to get the content
        response = self.tools['tavily_search_results_json'].invoke({"query": url})

        # Save content to state and add print for debugging
        state["content"] = response.get("content", "")
        if state["content"]:
            print("Content successfully fetched:")
            print(state["content"][:500])  # Print the first 500 characters for verification
        else:
            print("No content found for this URL.")

        return state


    def summarize_content(self, state):
        """Summarizes the article content (or first X characters?)."""
        
        return 

    def human_review(self, state):
        """Presents summaries and sentiment to the user and lets them decide if they want to keep it."""
        
        return 

"""
Consider how output will be received by next agent
"""

summary_prompt = """You are a smart and efficient research assistant working for a PR firm. 
You are being provided content from an article, and you need to generate a summary about this content. In your summary, only provide information
relative to the provided context. Do not hallucinate or make up any information. Only summarize what is provided."""

model = ChatOpenAI(model="gpt-4o")
search_params = {
    'filter': {
        'date': 'last_30_days',  
        'type': 'article'
    }
}
tool = TavilySearchResults(max_results=5, search_params=search_params)
search_agent = SummaryAgent(model, [tool], system=summary_prompt)
query = "Public perception of Angel Reese recent performances"
messages = [HumanMessage(content=query)]