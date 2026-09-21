"""
tools.py
--------
This file contains the external tools that our agents can use.

In LangChain, a "tool" is simply a Python function that we give to an agent.
The agent decides when to call the tool, what input to give it, and then
receives the output of the tool as an Observation.

We have two tools here:
1. web_search  → uses Tavily to search the internet
2. scrape_url  → visits a webpage and extracts the main text
"""

import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from langchain.tools import tool


# ------------------------------------------------------------------
# TOOL 1: Web Search (Tavily)
# ------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """
    Search the web for recent information about a topic.

    This tool is used by the Search Agent.
    The agent will call this tool when it needs fresh information
    from the internet.

    Args:
        query: The search query (example: "impact of AI on jobs 2026")

    Returns:
        A string containing the top search results (title + content + url)
    """
    # Create a Tavily client using the API key from environment variables
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Ask Tavily for the top 4 results. We keep the number small so the
    # agent does not get overloaded with too much text.
    response = client.search(query=query, max_results=4)

    # Format the results into a clean readable string
    formatted_results = []
    for i, result in enumerate(response.get("results", []), start=1):
        title = result.get("title", "No title")
        content = result.get("content", "No content")
        url = result.get("url", "No URL")
        formatted_results.append(
            f"Result {i}:\nTitle: {title}\nContent: {content}\nURL: {url}\n"
        )

    return "\n".join(formatted_results)


# ------------------------------------------------------------------
# TOOL 2: Scrape a single webpage
# ------------------------------------------------------------------
@tool
def scrape_url(url: str) -> str:
    """
    Visit a webpage and extract the main readable text.

    This tool is used by the Reader Agent.
    After the Search Agent finds useful links, the Reader Agent
    can call this tool to get the full content of a page.

    Args:
        url: The full URL of the page to scrape

    Returns:
        The main text content of the page (cleaned)
    """
    try:
        # Download the page. We set a User-Agent so some websites
        # do not block us as a bot.
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles so we only keep readable text
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Get all text and clean extra whitespace
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # Limit the length so the LLM does not receive an extremely long page
        if len(clean_text) > 4000:
            clean_text = clean_text[:4000] + "\n\n[Content truncated...]"

        return clean_text

    except Exception as e:
        return f"Failed to scrape the page. Error: {str(e)}"