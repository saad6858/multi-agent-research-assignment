"""
main.py
-------
Entry point of the multi-agent research system (Gemini Free Tier version).
"""

import os
import time
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain
)


def run_research_pipeline(topic: str):
    """
    Full multi-agent research pipeline with delays for free-tier rate limits.
    """

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


if __name__ == "__main__":
    topic = "The impact of AI on the job market in 2026"
    run_research_pipeline(topic)