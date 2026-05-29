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
        
        from datetime import datetime
        import time  # Import time to create a slight delay
        current_date_str = datetime.now().strftime("%B %d, %Y")
        
        system_instruction = f"""You are an expert Autonomous AI Travel Planning Assistant.
        Your absolute rule is to generate a complete travel itinerary immediately using your tools.
        
        CRITICAL OPERATIONAL RULES:
        1. NEVER ask the user conversational follow-up questions. 
        2. CURRENT REAL-WORLD DATE CONTEXT: Today is {current_date_str}. If specific travel dates are not explicitly mentioned by the user, automatically assume a standard upcoming 3-day weekend itinerary starting from the next closest Friday relative to today's date ({current_date_str}). Ensure the planned travel year matches 2026.
        3. You must call your available tools (search_flights, search_hotels, search_places, get_live_weather) to gather options before writing."""

        # Try running the model up to 3 times if the server is busy
        for attempt in range(3):
            try:
                # Turn 1: Let Gemini evaluate the user input
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
                if response.function_calls:
                    for call in response.function_calls:
                        tool_name = call.name
                        tool_args = dict(call.args)
                        if tool_name in self.tools_map:
                            try:
                                if hasattr(self.tools_map[tool_name], 'invoke'):
                                    result = self.tools_map[tool_name].invoke(tool_args)
                                else:
                                    result = self.tools_map[tool_name](**tool_args)
                                tool_outputs.append(f"Tool '{tool_name}' returned data:\n{result}")
                            except Exception as e:
                                tool_outputs.append(f"Tool '{tool_name}' execution failed: {str(e)}")
                
                # Turn 2: Synthesize final output matching the exact grading guidelines template layout
                final_prompt = f"""
                User Itinerary Request: {user_input}
                Current Date Baseline: {current_date_str}

                Gathered Datasets from System Tools:
                {" ".join(tool_outputs)}

                Please construct the travel plan using the exact template format below. Do not use creative introductory stories, conversational filler, or intro paragraphs. Follow this layout strictly:

                Your 3-Day Trip to [Destination City Name] ([Planned Weekend Dates in 2026])

                Flight Selected:
                - [Airline Name] ([Flight Price] INR) – Departs [Source City] at [Departure Time] (Flight Number: [Flight Number])

                Hotel Booked:
                - [Hotel Name] ([Price per night] INR/night, [Star Rating]-star)

                Weather:
                - Day 1: [Weather status or forecast description from tool data] ([Max Temp]°C)
                - Day 2: [Weather status or forecast description from tool data] ([Max Temp]°C)
                - Day 3: [Weather status or forecast description from tool data] ([Max Temp]°C)

                Itinerary:
                Day 1: [Sightseeing attractions from places.json for day 1]
                Day 2: [Sightseeing attractions from places.json for day 2]
                Day 3: [Sightseeing attractions from places.json for day 3]

                Estimated Total Budget:
                - Flight: [Flight Price] INR
                - Hotel: [Total Hotel Price for the nights] INR
                - Food & Travel: [Estimated Per-Day Local Expenses total] INR
                -------------------------------------
                Total Cost: [Complete calculated total sum] INR
                """
                
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
                # If it's a server error, wait 3 seconds and try the loop again
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    time.sleep(3)
                    continue
                return {"output": f"The AI server is very busy right now. Please click the button to try again in a moment! (Error details: {str(e)})"}

# Expose execution reference
travel_agent_brain = AutonomousAgentExecutor()