from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

def tavily_search(query):
    """
    Executes a web search using Tavily and returns a formatted markdown string of the top 5 results.
    Safely handles API connectivity errors.
    """

    try:
        response = client.search(
            query=query,
            max_results=5
        )
    except Exception as e:
        return f"Warning: Unable to fetch hotel web data due to a search engine error: {str(e)}"

    results = []

    search_records = response.get("results", [])

    for i, r in enumerate(search_records, 1):
        title   = r.get("title", "Unknown Page")
        url     = r.get("url", "#")
        snippet = r.get("content", "").strip()
        
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    if not results:
        return f"No relevant web search results found for the query: '{query}'."

    return "\n\n".join(results)
    
    
    
