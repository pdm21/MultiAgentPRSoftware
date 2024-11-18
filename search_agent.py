from dotenv import load_dotenv                                                                                                      # type: ignore
_ = load_dotenv()
from langgraph.graph import StateGraph, END                                                                                         # type: ignore
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage                                # type: ignore
from langchain_community.chat_models import ChatOpenAI                                                                  # type: ignore
from langchain_community.tools.tavily_search import TavilySearchResults # type: ignore
from langchain_openai import ChatOpenAI # type: ignore


# Specify how many results we want to get
tool = TavilySearchResults(max_results=5) 

# An annotated list of messages that we will add to over time.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class SearchAgent:
    # Parametrized by: the model to use, the tools to call, and the system message
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        
        # add nodes and edges for the graph
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        
        # compile graph and assign tools
        self.graph = graph.compile() 
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    # Takes in AgentState (list of messages) and calls the LLM on the messages plus the system prompt
    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
        
    # Are there any tool calls in the last message? If yes, there's action to take. If not, there is not.
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    # Takes the current AgentState (list of messages) and processes the last message, specifically looking for tool calls. 
    # It then invokes the corresponding tools and stores the results back in the state.
    def take_action(self, state: AgentState):
            tool_calls = state['messages'][-1].tool_calls       
            results = []
            for t in tool_calls:                                
                if not t['name'] in self.tools:                            
                    result = "bad tool name, retry"             
                else:               
                    result = self.tools[t['name']].invoke(t['args']) 
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result))) 
            return {'messages': results}