"""
main.py
-------
This is the entry point of the multi-agent research system.

It runs the four steps in sequence:
1. Search Agent finds information
2. Reader Agent digs deeper into useful pages
3. Writer creates the report
4. Critic reviews the report
"""

import os
from dotenv import load_dotenv

# Load the API keys from the .env file into environment variables
load_dotenv()

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)


import time

def run_research_pipeline(topic: str):
    print("\n" + "="*60)
    print("STEP 1 — Search Agent is working...")
    print("="*60)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "input": f"Find recent and reliable information about: {topic}"
    })
    search_text = search_result["output"]
    print("\n[Search Agent Output]\n")
    print(search_text)

    print("\nWaiting 45 seconds to respect free-tier rate limit...")
    time.sleep(45)

    print("\n" + "="*60)
    print("STEP 2 — Reader Agent is extracting deeper content...")
    print("="*60)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "input": (
            f"From the following search results about '{topic}', "
            f"choose the most useful URL and scrape its content.\n\n"
            f"{search_text}"
        )
    })
    reader_text = reader_result["output"]
    print("\n[Reader Agent Output]\n")
    print(reader_text)

    print("\nWaiting 45 seconds to respect free-tier rate limit...")
    time.sleep(45)

    print("\n" + "="*60)
    print("STEP 3 — Writer is creating the research report...")
    print("="*60)

    combined_research = (
        f"SEARCH RESULTS:\n{search_text}\n\n"
        f"DETAILED PAGE CONTENT:\n{reader_text}"
    )

    report = writer_chain.invoke({
        "topic": topic,
        "research": combined_research
    })
    print("\n[Final Research Report]\n")
    print(report)

    print("\nWaiting 30 seconds...")
    time.sleep(30)

    print("\n" + "="*60)
    print("STEP 4 — Critic is reviewing the report...")
    print("="*60)

    feedback = critic_chain.invoke({
        "report": report
    })
    print("\n[Critic Feedback]\n")
    print(feedback)

    print("\n" + "="*60)
    print("Pipeline finished successfully.")
    print("="*60)