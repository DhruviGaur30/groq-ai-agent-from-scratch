from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


# -------------------------------
# Custom Save-to-File Tool
# -------------------------------
def save_to_txt(data: str, filename: str = "research_output.txt"):
    """
    Save research data to a timestamped text file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"✅ Data successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file. Input should be a string.",
)


# -------------------------------
# Web Search Tool (DuckDuckGo)
# -------------------------------
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information. Input should be a search query string.",
)


# -------------------------------
# Wikipedia Tool
# -------------------------------
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = Tool(
    name="wikipedia",
    func=WikipediaQueryRun(api_wrapper=api_wrapper).run,
    description="Fetches a summary from Wikipedia for a given topic.",
)
