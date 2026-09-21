"""
agents.py
---------
This file defines the four main pieces of our multi-agent system:

1. Search Agent   → finds information on the web
2. Reader Agent   → reads / scrapes the most useful pages
3. Writer Chain   → writes a research report
4. Critic Chain   → reviews the report and gives feedback

We use Gemini as the brain of every agent and chain.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url


# ------------------------------------------------------------------
# Shared LLM (the brain)
# ------------------------------------------------------------------
# We create one Gemini model and reuse it for all agents.
# temperature=0 makes the answers more consistent and less random.
# This is important when we want reliable research behaviour.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# ------------------------------------------------------------------
# 1. SEARCH AGENT
# ------------------------------------------------------------------
def build_search_agent():
    """
    Creates the Search Agent.

    This agent only has access to the web_search tool.
    Its job is to find recent and relevant information about the topic.
    """
    # Official ReAct prompt from LangChain.
    # ReAct = Reason + Act.
    # The agent first thinks, then decides which tool to use,
    # then observes the tool result, then thinks again.
    prompt = hub.pull("hwchase17/react")

    # The agent only has one tool: web_search
    tools = [web_search]

    # create_react_agent connects the LLM + tools + thinking template
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # AgentExecutor is the engine that actually runs the loop:
    # Thought → Action → Observation → Thought → ... → Final Answer
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                 # shows the thinking steps (useful for screenshots)
        handle_parsing_errors=True,   # prevents crash if the model output is slightly messy
        max_iterations=4              # safety limit so it does not loop forever
    )

    return agent_executor


# ------------------------------------------------------------------
# 2. READER AGENT
# ------------------------------------------------------------------
def build_reader_agent():
    """
    Creates the Reader Agent.

    This agent has the scrape_url tool.
    After the Search Agent finds good links, this agent
    can open one of them and extract the full content.
    """
    prompt = hub.pull("hwchase17/react")
    tools = [scrape_url]

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    return agent_executor


# ------------------------------------------------------------------
# 3. WRITER CHAIN
# ------------------------------------------------------------------
# A chain is simpler than an agent.
# The Writer does not need tools. It only needs to read the research
# material and produce a clean report. So we use a normal chain.
writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional research writer. "
     "Write a clear, structured research report based on the given material. "
     "Use headings and keep the tone formal but easy to understand."),
    ("human",
     "Topic: {topic}\n\n"
     "Research Material:\n{research}\n\n"
     "Write a complete research report.")
])

# This is a simple pipeline: prompt → LLM → text
writer_chain = writer_prompt | llm | StrOutputParser()


# ------------------------------------------------------------------
# 4. CRITIC CHAIN
# ------------------------------------------------------------------
critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict research critic. "
     "Evaluate the report for clarity, structure, usefulness and possible missing points. "
     "Give a score out of 10 and short improvement suggestions."),
    ("human",
     "Here is the research report:\n\n{report}\n\n"
     "Provide your critical feedback.")
])

critic_chain = critic_prompt | llm | StrOutputParser()