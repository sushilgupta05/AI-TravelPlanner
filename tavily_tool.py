from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()


# https://www.tavily.com/ 
# Signup and login, On dashboard- > under api keys you will see the default key.
# Use that or click on + to create new one. Then save it in .env file

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

# test it
#################################
# response = client.search(
    # query="Best hotels in Dubai"
# )

# print(response)

####################################



def tavily_search(query):
    """
    Executes a web search using Tavily and returns a formatted markdown string of the top 5 results.
    Safely handles API connectivity errors.
    """
    # 1. Wrap the network call in a try-except block to handle API down/timeout events
    try:
        response = client.search(
            query=query,
            max_results=5
        )
    except Exception as e:
        # Return a clean error message text to the LLM agent instead of crashing the program
        return f"Warning: Unable to fetch hotel web data due to a search engine error: {str(e)}"

    results = []

    # 2. Use .get() defensively in case the response structure unexpected changes or lacks "results"
    search_records = response.get("results", [])

    for i, r in enumerate(search_records, 1):
        title   = r.get("title", "Unknown Page")
        url     = r.get("url", "#")
        snippet = r.get("content", "").strip()
        
        # Keep only the first 300 characters to avoid wall-of-text crashing the LLM's context window
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    # 3. Handle cases where the API call succeeds but finds absolutely zero web results
    if not results:
        return f"No relevant web search results found for the query: '{query}'."

    return "\n\n".join(results)
    
    
    
