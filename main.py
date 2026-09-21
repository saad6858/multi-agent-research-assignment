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


def run_research_pipeline(topic: str):
    """
    Full multi-agent research pipeline.
    """

    print("\n" + "="*60)
    print("STEP 1 — Search Agent is working...")
    print("="*60)

    # Build and run the Search Agent
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "input": f"Find recent and reliable information about: {topic}"
    })

    # The final answer of the agent is stored in the "output" key
    search_text = search_result["output"]
    print("\n[Search Agent Output]\n")
    print(search_text)


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


    print("\n" + "="*60)
    print("STEP 3 — Writer is creating the research report...")
    print("="*60)

    # Combine both sources of information for the writer
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
    # Change this topic to research anything else
    topic = "The impact of AI on the job market in 2026"
    run_research_pipeline(topic)