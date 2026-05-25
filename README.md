# 🌍 AI Travel Booking System: Multi-Agent Concierge

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-000000?style=for-the-badge&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_LLaMA_3.3-f55036?style=for-the-badge)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)

An advanced, multi-agent AI travel planner that searches live flights, fetches hotel recommendations, and builds comprehensive, day-by-day travel itineraries. Built with **Streamlit**, orchestrated by **LangGraph**, powered by **Groq (LLaMA 3.3)**, and backed by a persistent **PostgreSQL (Supabase)** database for conversational memory.

---

## ✨ Key Features

* **🤖 Multi-Agent Architecture:** Four specialized AI agents (Flight, Hotel, Itinerary, and Final Compilation) work sequentially to process complex travel requests.
* **✈️ Live Flight Data:** Integrates with the **AviationStack API** to fetch real-time airline schedules, routes, and flight statuses based on user-defined parameters.
* **🏨 Real-time Hotel Search:** Uses the **Tavily Search API** to scour the web for top-rated hotels matching specific budget constraints and location preferences.
* **🧠 Persistent Memory:** Saves all generated travel plans and chat histories to a **Supabase PostgreSQL** database using LangGraph Checkpointers. Users can access past trips seamlessly from the sidebar.
* **📄 PDF Export:** Generates a beautifully formatted, professional Travel Dossier that users can download as a local PDF directly from the UI.
* **🎨 Custom Dark Theme UI:** Features a sleek, modern Streamlit interface with hero banners, destination quick-picks, and dynamic pipeline status tracking.

---

## 🏗️ System Architecture

The application relies on a LangGraph `StateGraph` where a user query is passed through specialized nodes:

1. **Flight Agent (`flight_agent`):** Extracts departure/arrival IATA codes and dates via structured LLM output, fetching live flight data from AviationStack.
2. **Hotel Agent (`hotel_agent`):** Extracts destination and preferences, fetching live web results for accommodations via Tavily Search.
3. **Itinerary Agent (`itinerary_agent`):** Takes the output of the first two agents and drafts a highly detailed, geographically logical day-by-day itinerary.
4. **Final Agent (`final_agent`):** Compiles the raw data into a visually stunning, formatted Markdown dossier with estimated budget breakdowns.

---
### API needed
You will need API keys for the following services (all offer generous free tiers):
* **Groq** (LLM inference)
* **Tavily** (Web search)
* **AviationStack** (Flight data)
* **Supabase** (PostgreSQL Database)


## 📁 Project Structure

```text
├── frontend.py          # Streamlit UI, custom CSS styling, and in-memory PDF generation
├── main.py              # LangGraph multi-agent pipeline, prompts, and Supabase DB connection
├── flight_tool.py       # AviationStack API integration logic
├── tavily_tool.py       # Tavily Web Search API integration logic
├── requirements.txt     # Python dependencies (Streamlit, LangGraph, psycopg, etc.)
├── packages.txt         # Linux system dependencies for Cloud PDF rendering
├── .gitignore           # Ignored files (.env, __pycache__, etc.)
└── README.md            # Project documentation
---
