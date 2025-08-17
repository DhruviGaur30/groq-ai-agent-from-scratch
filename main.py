# Import required libraries
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI  # OpenAI-compatible wrapper
from tools import search_tool, wiki_tool, save_tool

import os

# Load environment variables from .env file:
# Make sure you have GROQ_API_KEY in your .env file like:
# GROQ_API_KEY=your_groq_api_key_here
load_dotenv()

# Define structured response schema:
# Pydantic ensures the AI output follows this structure strictly
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize Groq LLM via LangChain:
# Groq uses an OpenAI-compatible API, so we can use ChatOpenAI.
# Just point it to Groq’s API endpoint and pass the key.
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",  # Recommended replacement
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0
)


# Set up output parser:
# This parser will enforce that the output matches ResearchResponse format
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# Create the system + human prompt
# The system role defines how the AI should behave
# We also insert the format instructions for structured output
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),   # conversation memory placeholder
        ("human", "{query}"),                # user query goes here
        ("placeholder", "{agent_scratchpad}")# space for agent's tool reasoning
    ]
).partial(format_instructions=parser.get_format_instructions())

# Register the tools
# ------------------------------
# These are helper functions your agent can call when needed
tools = [search_tool, wiki_tool, save_tool]

# Create the Agent
# ------------------------------
# The agent is responsible for:
# 1. Deciding which tool to call
# 2. Executing the tool
# 3. Formatting the final structured response
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Create the Agent Executor
# ------------------------------
# AgentExecutor actually runs the loop:
# - Takes input
# - Lets the agent reason about tools
# - Collects output
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the program
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

# Try parsing the structured response:
# The parser ensures the response matches our ResearchResponse model.
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print("\n✅ Structured Research Response:")
    print(structured_response)
except Exception as e:
    print("❌ Error parsing response:", e)
    print("Raw Response - ", raw_response)
