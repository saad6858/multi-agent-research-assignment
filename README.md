```markdown
# Multi-Agent Research System (Gemini)

Multi-agent research pipeline built during the Bano Qabil Agentic AI program (Batch BQL6-LR).

**Student:** Saad Maqbool  
**Campus:** Lytton Road, Lahore

## Overview

This project implements a multi-agent research system using Gemini Free Tier. The system:

1. Searches the web for information on a given topic
2. Extracts deeper content from useful pages
3. Writes a structured research report
4. Critiques the report and provides feedback

## Architecture

- **Search Agent** → Uses Tavily Search API to find recent information
- **Reader Agent** → Scrapes and extracts content from selected URLs
- **Writer Chain** → Generates a full research report
- **Critic Chain** → Reviews the report and gives quality feedback

## Technologies

- LangChain (agent orchestration)
- Google Gemini (LLM via langchain-google-genai)
- Tavily (Search API)
- BeautifulSoup (web scraping)
- python-dotenv

## Project Structure

- `main.py` — Entry point and pipeline orchestration
- `agents.py` — Search Agent, Reader Agent, Writer Chain, Critic Chain
- `tools.py` — web_search and scrape_url tools
- `requirements.txt`
- `.env.example`
- `README.md`

## Setup

1. Clone the repository

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Create your environment file:

```bash
cp .env.example .env
```

Add your keys:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

4. Run the pipeline:

```bash
python main.py
```

## Workflow

1. User provides a research topic
2. Search Agent gathers web results using Tavily
3. Reader Agent scrapes the most relevant pages
4. Writer Chain produces the research report
5. Critic Chain evaluates the report

## Notes

This implementation is designed for Gemini Free Tier. Short delays are added between major steps to respect rate limits. The focus is on clear agent roles, tool usage, and an end-to-end research pipeline.
```