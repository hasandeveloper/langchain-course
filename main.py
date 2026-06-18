from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

class Source(BaseModel):
    ''' Schema for a source used by the agent '''
    url:str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    ''' Schema for the response from the agent '''
    answer: str = Field(description="The answer from the agent")
    sources: List[Source] = Field(default_factory=list, description="The sources used by the agent")

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query, include_domains=["linkedin.com"])


llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="list the linkedin jobs for ai engineer in uae")})
    print(result)


if __name__ == "__main__":
    main()


# very important
# https://docs.langchain.com/oss/python/langchain/structured-output