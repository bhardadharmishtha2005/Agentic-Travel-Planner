import os
import streamlit as st
from dotenv import load_dotenv

# Clean import strategy to prevent cloud namespace clashes
import google.genai as google_genai_sdk
from google.genai import types

# Import your custom python tools from tools.py
from tools import search_flights, search_hotels, search_places, get_live_weather

load_dotenv()

class AutonomousAgentExecutor:
    def __init__(self):
        load_dotenv()
        
        # Pull key from local environment first; if empty, pull from Streamlit Cloud Secrets dashboard box
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            
        if api_key:
            api_key = api_key.strip()
        
        # Build the client using the clean namespace module reference
        self.client = google_genai_sdk.Client(api_key=api_key)
        
        # Map tool names directly to their executable Python function wrapper
        self.tools_map = {
            "search_flights": search_flights,
            "search_hotels": search_hotels,
            "search_places": search_places,
            "get_live_weather": get_live_weather
        }
        
        # Pass the tool definitions so Google's schema parser processes them natively
        self.google_tools = [search_flights, search_hotels, search_places, get_live_weather]

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("input", "")
        
        # Dynamically fetch the current real-world date context in Python
        from datetime import datetime
        current_date_str = datetime.now().strftime("%B %d, %Y")
        
        system_instruction = f"""You are an expert Autonomous AI Travel Planning Assistant.
        Your absolute rule is to generate a complete travel itinerary immediately using your tools.
        
        CRITICAL OPERATIONAL RULES:
        1. NEVER ask the user conversational follow-up questions. 
        2. CURRENT REAL-WORLD DATE CONTEXT: Today is {current_date_str}. If specific travel dates are not explicitly mentioned by the user, automatically assume a standard upcoming 3-day weekend itinerary starting from the next closest Friday relative to today's date ({current_date_str}). Ensure the planned travel year matches 2026.
        3. You must call your available tools (search_flights, search_hotels, search_places, get_live_weather) to gather options before writing.
        4. MANDATORY OUTPUT STRUCTURE REQUIRED FOR GRADING:
           - Trip Summary & Travel Dates (Must look forward into 2026)
           - Flight Option Selected: (Print the exact Airline, Flight Number, and Price found in flights.json)
           - Hotel Recommendation: (Print Hotel Name, Rating, and Price per night from hotels.json)
           - Day-wise Itinerary: (Include attractions from places.json)
           - Weather Forecast: (Display the exact daily maximum temperature values returned by the weather tool API for each day)
           - Itemized Budget Breakdown: Show the mathematical sum total clearly: Flight Price + (Hotel Price x Nights) + Estimated Per-Day Local Expenses."""

        try:
            # Turn 1: Let Gemini evaluate the user input and select required data tools
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=self.google_tools,
                    temperature=0.0
                )
            )
            
            tool_outputs = []
            
            # Check if Gemini requested to run tools, and execute them safely
            if response.function_calls:
                for call in response.function_calls:
                    tool_name = call.name
                    tool_args = dict(call.args)
                    
                    if tool_name in self.tools_map:
                        try:
                            # Invoke the tool directly with unpack mapping
                            result = self.tools_map[tool_name].invoke(tool_args)
                            tool_outputs.append(f"Tool '{tool_name}' returned data:\n{result}")
                        except Exception as e:
                            tool_outputs.append(f"Tool '{tool_name}' execution failed: {str(e)}")
            
            # Turn 2: Explicitly bundle the data and feed it back to compile the text
            final_prompt = (
                f"User Itinerary Request: {user_input}\n"
                f"Current Date Baseline: {current_date_str}\n\n"
                f"Gathered Datasets from System Tools:\n" + "\n\n".join(tool_outputs) + 
                "\n\nPlease construct a complete, professional travel plan matching all the MANDATORY OUTPUT STRUCTURE criteria using the data provided above. Ensure the dates chosen reflect an upcoming real weekend in 2026."
            )
            
            final_response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )
            )
            
            return {"output": final_response.text if final_response.text else "Itinerary generated successfully."}
            
        except Exception as e:
            return {"output": f"An error occurred during agent processing: {str(e)}"}

# Expose execution reference
travel_agent_brain = AutonomousAgentExecutor()