'''
# pip install langgraph langchain langchain-openai langchain-groq langchain-community langchain-tavily psycopg[binary] psycopg_pool python-dotenv tavily-python pip install requests streamlit

# install PostgresSql and create database
CREATE DATABASE langgraph_memory;  ( or open pgadmin4 and create database there )
'''
# LangGraph Multi-Agent Travel Booking System with Long-Term Memory

# main.py

import os
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator


from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tavily_tool import tavily_search
from flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()


# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Schema

# For flight agent
class FlightRoutingSchema(BaseModel):
    dep_iata: str = Field(
        description="The 3-letter IATA code for the departure airport."
    )
    arr_iata: str = Field(
        description="The 3-letter IATA code for the arrival (destination) airport."
    )
    travel_dates: str = Field(
        description="The extracted travel dates, duration, or 'Dates not specified'."
    )

# for hotel agent
class HotelSearchSchema(BaseModel):
    destination: str = Field(
        description="The final destination city or region where the user will stay."
    )
    preferences: str = Field(
        description="Specific hotel preferences (e.g., 'luxury', 'budget', 'under ₹50,000', 'near beach'). If none mentioned, output 'best rated'."
    )

# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

# Flight Agent
def flight_agent(state: TravelState):
    user_query = state["user_query"]
    
    # 1. Bind the Pydantic schema to the LLM
    structured_llm = llm.with_structured_output(FlightRoutingSchema)
    
    # 2. The Enriched Prompt (Business Logic & Reasoning)
    routing_prompt = """
    You are an expert travel routing AI. Your job is to analyze the user's travel request and determine the exact departure and arrival airports, along with any mentioned dates.

    Strict Routing Rules:
    1. DEPARTURE: If a city is explicitly mentioned (e.g., Mumbai), use its primary IATA code (BOM). If NO departure location is mentioned, default to 'DEL' (Delhi) as the standard hub.
    2. ARRIVAL: If the destination is a specific city, use its primary airport code (e.g., Tokyo -> HND). If the destination is a country, map it to that country's busiest international gateway (e.g., Japan -> NRT, France -> CDG).
    3. DATES: Extract any mentioned dates, durations, or months. If none are mentioned, output 'Dates not specified'.
    """
    
    # Safe fallbacks
    dep_iata = "DEL"
    arr_iata = "NRT"
    travel_dates = "Dates not specified"
    
    try:
        # 3. Invoke the LLM with BOTH the enriched reasoning prompt and the strict Pydantic schema
        result = structured_llm.invoke([
            SystemMessage(content=routing_prompt),
            HumanMessage(content=f"User Query: {user_query}")
        ])
        
        # Access the structured Pydantic fields directly
        dep_iata = result.dep_iata.strip().upper()
        arr_iata = result.arr_iata.strip().upper()
        travel_dates = result.travel_dates.strip()
        
    except Exception as e:
        print(f"Error parsing routing schema, applying fallbacks: {e}")

    # 4. Call the flight tool safely
    flight_data = search_flights(dep_iata, arr_iata)
    
    return {
        "flight_results": f"📅 Travel Dates: {travel_dates}\n✈️ Route Resolved: {dep_iata} ➔ {arr_iata}\n\n{flight_data}",
        "messages": [
            AIMessage(content=f"Flight results fetched for route {dep_iata} to {arr_iata}. Dates: {travel_dates}")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1 
    }

# Hotel Agent
def hotel_agent(state: TravelState):
    user_query = state["user_query"]
    
    # 1. Bind the Pydantic schema to the LLM
    structured_llm = llm.with_structured_output(HotelSearchSchema)
    
    # 2. The Enriched Prompt (Business Logic & Reasoning)
    hotel_prompt = """
    You are an expert travel accommodation AI. Analyze the user's request and extract the destination and any specific hotel preferences.

    Extraction Rules:
    1. DESTINATION: Identify the final city or region where the user needs a hotel. If the user mentions a country (e.g., "trip to Japan"), pick the primary tourist city (e.g., "Tokyo") unless specified otherwise. If no location is found, default to 'Delhi'.
    2. PREFERENCES: Extract any keywords related to the stay. Look for budget constraints (e.g., "cheap", "under 2 lakhs"), star ratings ("5-star"), or styles ("romantic", "family-friendly"). If none are mentioned, output 'best rated'.
    """
    
    # Safe fallbacks
    destination = "Delhi"
    preferences = "best rated"
    
    try:
        # 3. Invoke the LLM with BOTH reasoning rules and strict schema
        result = structured_llm.invoke([
            SystemMessage(content=hotel_prompt),
            HumanMessage(content=f"User Query: {user_query}")
        ])
        
        # Access the structured Pydantic fields
        destination = result.destination.strip()
        preferences = result.preferences.strip()
        
    except Exception as e:
        print(f"Error parsing hotel schema, applying fallbacks: {e}")

    # 4. Build a highly targeted search string for Tavily
    clean_search_query = f"Top {preferences} hotels to stay in {destination} with prices and recent reviews"
    
    # 5. Call the Tavily tool
    hotel_results = tavily_search(clean_search_query)

    return {
        "hotel_results": f"🏨 Location: {destination}\n🛏️ Preferences: {preferences}\n\n{hotel_results}",
        "messages": [
            AIMessage(content=f"Hotel information fetched for {destination} matching preferences: {preferences}")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1 
    }

# Itinerary Agent
def itinerary_agent(state: TravelState):
    user_query = state['user_query']
    flight_results = state['flight_results']
    hotel_results = state['hotel_results']

    system_prompt = """
    You are an elite, highly knowledgeable luxury travel planner.
    Your job is to design a rich, expansive, and highly readable day-by-day itinerary.

    STRICT RULES FOR STRUCTURE AND QUANTITY:
    1. CHAIN OF THOUGHT REASONING: Before writing, think step-by-step about the geography of the city. Do not make the user cross the city multiple times in one day. Group nearby attractions logically.
    2. EXTREME DETAIL: Each time block (Morning, Afternoon, Evening) must be a robust, highly detailed paragraph (at least 3 to 4 sentences each). 
    3. HISTORICAL & CULTURAL CONTEXT: When suggesting a landmark, provide 1-2 sentences of fascinating historical background or cultural significance.
    4. GASTRONOMY: Include specific local food recommendations for meals (e.g., "Try authentic Takoyaki at a local street stall").
    5. FORMATTING: Use clean Markdown. Use '### Day 1: [Theme]' for headers. 

    6. VISUAL SPACING (CRITICAL): You MUST use single line breaks (\n) between Morning, Afternoon, and Evening blocks.
    
    7. THE EXACT REQUIRED FORMAT FOR EVERY DAY:
       ### Day [X]: [Engaging Theme for the Day]
       
       **🌅 Morning:**
       [Write 3-4 rich sentences here. Include historical context and specific breakfast spots.]
       
       **☀️ Afternoon:**
       [Write 3-4 rich sentences here. Include logistics, sightseeing, and lunch.]
       
       **🌙 Evening:**
       [Write 3-4 rich sentences here. Include nightlife, dinner, and relaxation.]
       
       **💰 Estimated Cost for the specific day: ** [Provide cost]
    """

    human_prompt = f"""
    User Request: "{user_query}"

    --- SEARCH DATA ---
    FLIGHTS: {flight_results}
    HOTELS: {hotel_results}
    -------------------
    
    Draft the highly detailed, perfectly spaced day-by-day itinerary now.
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        itinerary_content = response.content
    except Exception as e:
        print(f"Error generating itinerary: {e}")
        itinerary_content = f"⚠️ I encountered an issue while drafting your itinerary: {str(e)}."

    return {
        "itinerary": itinerary_content,
        "messages": [AIMessage(content="Detailed itinerary synthesized.")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Final Response Agent
def final_agent(state: TravelState):
    flight_results = state['flight_results']
    hotel_results = state['hotel_results']
    itinerary = state['itinerary']
    user_query = state['user_query']

    system_prompt = """
    You are a world-class travel concierge. Your goal is to produce a massive, deeply detailed, and visually stunning travel dossier. 
    
    TONE: Warm, highly professional, empathetic, and exceptionally knowledgeable. Speak directly to the user (e.g., "I have curated...", "You will love...").
    STRICT FORMATTING RULES (CRITICAL):
    - SPACING: You must use double line breaks (\n\n) between all paragraphs and sections to prevent bulky text.
    - TABLES: You MUST use standard Markdown table syntax. You must include a line break after every single row so the table does not collapse into a single line.

    YOU MUST FOLLOW THIS EXACT MARKDOWN STRUCTURE:

    # 🌍 Your Curated Travel Dossier

    *Write a warm, 2-paragraph welcome message...*

    ## 📜 Destination Deep-Dive & History
    
     *Provide a rich, 4-paragraph overview of the destination's history, culture, and what makes it unique.*

    ## ✈️ Flight Strategy & Logistics
    
     *Explain the best flight options from the provided data. Give practical advice on navigating the arrival airport and getting to the city center.*

    ## 🏨 Your Stay
    
     *Detail the best 3 selected hotel from the data. Explain the neighborhood vibe and why it is a great choice. Also tell the cost of it*

    ## 🗺️ The Immersive Day-by-Day Itinerary
    
    *CRITICAL: Paste the exact, fully detailed day-by-day itinerary provided in the data. Preserve all line breaks (\n\n), bolding, and spacing exactly as generated.*

    ## 🎭 Cultural Awareness & Etiquette
    
    *Provide 8-9 bullet points on local customs, tipping culture, dress codes, safety tips, or important phrases to know.*

    ## 💰 Estimated Budget Breakdown

        All estimated costs should be realistic price ranges derived from the hotel_agent response and current travel pricing data. Ensure totals are internally consistent and reflect current market rates.
    
    | Category | Estimated Cost | Notes |
    |---|---|---|
    | ✈️ Flights | [Cost] | [Note] |
    | 🏨 Hotel | [Cost] | [Note] |
    | 🍜 Food | [Cost] | [Note] |
    | 🎟️ Activities | [Cost] | [Note] |
    | 🛍️ Buffer | [Cost] | [Note] |

    ## 💡 Concierge Sign-Off
    
    *A warm closing statement... along with wishing them a safe journey with emoji*
    """

    final_prompt = f"""
    Original Request: "{user_query}"

    --- DATA BLOCK 1: FLIGHTS ---
    {flight_results}
    
    --- DATA BLOCK 2: HOTELS ---
    {hotel_results}
    
    --- DATA BLOCK 3: FULL ITINERARY (DO NOT SUMMARIZE, PRESERVE ALL SPACING) ---
    {itinerary}
    
    Generate the complete dossier now following the exact Markdown structure and spacing requested.
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=final_prompt)
        ])
        final_content = response.content
    except Exception as e:
        print(f"Error in final agent compilation: {e}")
        final_content = f"### ✈️ Curated Travel Summary\n\n{flight_results}\n\n{hotel_results}\n\n### 🗓️ Itinerary\n\n{itinerary}"

    return {
        "messages": [AIMessage(content=final_content)],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

app = graph.compile()