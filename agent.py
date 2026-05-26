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
        
        system_instruction = """You are an expert Autonomous AI Travel Planning Assistant.
        Your absolute rule is to generate a complete travel itinerary immediately using your tools.
        
        CRITICAL OPERATIONAL RULES:
        1. NEVER ask the user conversational follow-up questions. Assume a standard upcoming 3-day weekend if dates are absent.
        2. You must call your available tools (search_flights, search_hotels, search_places, get_live_weather) to gather options before writing.
        3. MANDATORY OUTPUT STRUCTURE REQUIRED FOR GRADING:
           - Trip Summary & Travel Dates
           - Flight Option Selected: (Must print the exact Airline, Flight Number, and Price found in flights.json)
           - Hotel Recommendation: (Must print Hotel Name, Rating, and Price per night from hotels.json)
           - Day-wise Itinerary: (Include attractions from places.json)
           - Weather Forecast: (Must display the exact daily maximum temperature values returned by the weather tool API for each day)
           - Itemized Budget Breakdown: Show the mathematical sum total clearly: Flight Price + (Hotel Price x Nights) + Estimated Per-Day Local Expenses."""

        # Turn 1: Let the model evaluate the user input and select required data tools
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
                        result = self.tools_map[tool_name].invoke(tool_args)
                        tool_outputs.append(f"Tool '{tool_name}' returned data:\n{result}")
                    except Exception as e:
                        tool_outputs.append(f"Tool '{tool_name}' execution failed: {str(e)}")
            
            # Turn 2: Feed all gathered data back to Gemini to compile the final text itinerary
            final_prompt = (
                f"User Itinerary Request: {user_input}\n\n"
                f"Gathered Datasets from System Tools:\n" + "\n\n".join(tool_outputs) + 
                "\n\nPlease construct a complete, professional day-by-day travel plan using only the options provided above."
            )
            
            final_response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=final_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )
            )
            return {"output": final_response.text}
            
        return {"output": response.text if response.text else "No response generated. Please try again."}

# Expose execution reference
travel_agent_brain = AutonomousAgentExecutor()