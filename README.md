# Agentic AI-Based Travel Planning Assistant

An autonomous, data-driven travel planning ecosystem built using **LangChain**, **Gemini 2.5 Flash**, and **Streamlit**. The application leverages native Tool Calling reasoning frameworks to dynamically query local aviation, hospitality, and landmark datasets alongside real-time weather forecasts to deliver comprehensive, optimized, and itemized travel itineraries.

---

## 🚀 Key Features

- **Autonomous Agentic Execution**: Utilizing Google Gemini's native tool-calling layers with multi-step reasoning capabilities to interpret raw user prompts without needing repetitive conversational follow-ups.
- **Dynamic Local Data Parsing**: Real-time evaluation and filtering of custom JSON databases for localized flight details, hotel choices, and point-of-interest mapping based on budget and star-rating structures.
- **Live API Integration**: Native integration with the free Open-Meteo Weather API to predict exact maximum daily temperature expectations for chosen vacation periods.
- **Automated Cost Summarization**: Calculates complete financial breakdowns factoring in exact flight ticketing costs, cumulative multi-night hotel stay tariffs, and local per-day allowances.

---

## 🛠️ System Architecture

The workflow follows a robust two-turn tool execution pattern:
1. **Turn 1 (Intent Parsing)**: The user provides basic source, destination, and budget metrics via a Streamlit interface. The Gemini engine evaluates the prompt and determines which functional data tool extensions are required.
2. **Execution Loop**: Selected tool scripts read structural JSON formats (`flights.json`, `hotels.json`, `places.json`) and call the external live weather endpoints.
3. **Turn 2 (Synthesis)**: All structured tool response results are fed back into the model to output a pristine, day-wise trip itinerary mapping.

---

## 🗂️ Project Structure

```text
├── .gitignore             # Git exclusion mapping configuration (venv, .env)
├── agent.py               # Core Autonomous Agent Executor brain logic
├── app.py                 # Streamlit UI layout and presentation dashboard
├── tools.py               # Functional implementation of dataset & API extraction tools
├── flights.json           # JSON Database storing flight routes, numbers, and pricing
├── hotels.json            # JSON Database containing hotel options, budgets, and ratings
├── places.json            # JSON Database including point-of-interest tourist spots
├── requirements.txt       # Unified Python library dependencies setup reference
└── README.md              # Project documentation manual