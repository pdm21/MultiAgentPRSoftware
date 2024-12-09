from langgraph.graph import StateGraph, END
from langgraph.tool import Tool
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
import requests
import json
from typing import List, Dict


# Tool 1: Perplexity API Tool
class PerplexityTool(Tool):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def invoke(self, args: Dict):
        athlete = args["athlete"]
        query = f"Find the 10 most recent articles about {athlete}."
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.2,
            "max_tokens": 1000,
            "stream": False
        }
        response = requests.post(self.url, json=payload, headers=self.headers)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return [line.strip() for line in content.split("\n") if line.strip()]


# Tool 2: Summarization Tool
class SummarizationTool(Tool):
    def invoke(self, args: Dict):
        links = args["links"]
        summaries = {link: f"Summary for {link}" for link in links}  # Simulated summary
        return summaries


# Tool 3: Human-in-the-Loop Tool
class HumanFilterTool(Tool):
    def invoke(self, args: Dict):
        summaries = args["summaries"]
        kept_links = {}
        for link, summary in summaries.items():
            print(f"\nLink: {link}\nSummary: {summary}")
            user_input = input("Keep this article? (y/n): ").strip().lower()
            if user_input == "y":
                kept_links[link] = summary
        return kept_links


# Tool 4: Sentiment Analysis Tool
class SentimentTool(Tool):
    def invoke(self, args: Dict):
        summaries = args["summaries"]
        sentiment = {}
        for link, summary in summaries.items():
            sentiment[link] = "positive" if "good" in summary.lower() else "negative"
        return sentiment


# Tool 5: Write-Up Tool
class WriteUpTool(Tool):
    def invoke(self, args: Dict):
        sentiments = args["sentiments"]
        positive_content = []
        negative_content = []

        for link, sentiment in sentiments.items():
            if sentiment == "positive":
                positive_content.append(link)
            else:
                negative_content.append(link)

        positive_writeup = f"Positive Articles:\n" + "\n".join(positive_content)
        negative_writeup = f"Negative Articles:\n" + "\n".join(negative_content)

        return {
            "positive": positive_writeup,
            "negative": negative_writeup
        }


# Define the Graph
class AthletePRAgent:
    def __init__(self, api_key: str):
        self.graph = StateGraph()
        self.perplexity_tool = PerplexityTool(api_key)
        self.summarization_tool = SummarizationTool()
        self.human_filter_tool = HumanFilterTool()
        self.sentiment_tool = SentimentTool()
        self.writeup_tool = WriteUpTool()

        # Define the graph flow
        self.graph.add_node("perplexity", self.perplexity_tool.invoke)
        self.graph.add_node("summarization", self.summarization_tool.invoke)
        self.graph.add_node("human_filter", self.human_filter_tool.invoke)
        self.graph.add_node("sentiment", self.sentiment_tool.invoke)
        self.graph.add_node("writeup", self.writeup_tool.invoke)

        self.graph.add_edge("perplexity", "summarization")
        self.graph.add_edge("summarization", "human_filter")
        self.graph.add_edge("human_filter", "sentiment")
        self.graph.add_edge("sentiment", "writeup")

        self.graph.set_entry_point("perplexity")

    def run(self, athlete: str):
        context = {"athlete": athlete}
        return self.graph.execute(context)


# Example Usage
if __name__ == "__main__":
    api_key = "your-perplexity-api-key"
    agent = AthletePRAgent(api_key)
    result = agent.run("Lionel Messi")
    
    print("\nMedia Sentiment Report:")
    print(f"A report has been generated on your athlete.")
    print(f"Here are the positive things that are being said: {result['positive']}")
    print(f"Here are the negative things: {result['negative']}")
