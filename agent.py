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
    """
    An Agentic AI System built using the Google GenAI SDK that autonomously 
    orchestrates travel planning by executing a multi-turn tool calling workflow.
    """
    def __init__(self):
        load_dotenv()
        
        # Pull API key from local environment or Streamlit Secrets dashboard securely
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            
        if api_key:
            api_key = api_key.strip()
        
        # Initialize the Google GenAI client reference
        self.client = google_genai_sdk.Client(api_key=api_key)
        
        # Map tool keys directly to their executable Python functions
        self.tools_map = {
            "search_flights": search_flights,
            "search_hotels": search_hotels,
            "search_places": search_places,
            "get_live_weather": get_live_weather
        }
        
        # List of tool signatures passed to the model for functional routing
        self.google_tools = [search_flights, search_hotels, search_places, get_live_weather]

    def invoke(self, inputs: dict) -> dict:
        """
        Executes a two-turn ReAct / Tool-Calling reasoning loop to interpret 
        user travel requests, call tools, and synthesize structured plans.
        """
        user_input = inputs.get("input", "")
        
        from datetime import datetime
        import time  # Used to build recovery delay loops for server stability
        current_date_str = datetime.now().strftime("%B %d, %Y")
        
        # High-quality system instructions ensuring strict compliance with operational rules
        system_instruction = f"""You are an expert Autonomous AI Travel Planning Assistant.
        Your absolute rule is to generate a complete travel itinerary immediately using your tools.
        
        CRITICAL OPERATIONAL RULES:
        1. NEVER ask the user conversational follow-up questions. Always complete the plan in one run.
        2. CURRENT REAL-WORLD DATE CONTEXT: Today is {current_date_str}. If specific travel dates are not explicitly mentioned by the user, automatically assume a standard upcoming 3-day weekend itinerary starting from the next closest Friday relative to today's date ({current_date_str}). Ensure the planned travel year matches 2026.
        3. You must call your available tools (search_flights, search_hotels, search_places, get_live_weather) to gather real-world options before writing the itinerary."""

        # Robust try-except error handling loop with automatic multi-attempt retries
        for attempt in range(3):
            try:
                # =====================================================================
                # TURN 1: Autonomous Intent Analysis & Tool Calling Decision
                # =====================================================================
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_input,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=self.google_tools,
                        temperature=0.0  # Greedy decoding for high tool accuracy
                    )
                )
                
                tool_outputs = []
                # Execute called functions programmatically if the model chose to use them
                if response.function_calls:
                    for call in response.function_calls:
                        tool_name = call.name
                        tool_args = dict(call.args)
                        if tool_name in self.tools_map:
                            try:
                                # Safe execution check for LangChain-wrapped structures or raw callables
                                if hasattr(self.tools_map[tool_name], 'invoke'):
                                    result = self.tools_map[tool_name].invoke(tool_args)
                                else:
                                    result = self.tools_map[tool_name](**tool_args)
                                tool_outputs.append(f"Tool '{tool_name}' returned data:\n{result}")
                            except Exception as e:
                                tool_outputs.append(f"Tool '{tool_name}' execution failed: {str(e)}")
                
                # =====================================================================
                # TURN 2: Data Synthesis & Strict Template Formatting Response
                # =====================================================================
                final_prompt = f"""
                User Itinerary Request: {user_input}
                Current Date Baseline: {current_date_str}

                Gathered Datasets from System Tools:
                {" ".join(tool_outputs)}

                Please construct the travel plan using the exact syllabus template format below. Do not use creative introductory stories or conversational paragraphs. Follow this layout strictly:

                Your 3-Day Trip to [Destination City Name] ([Planned Weekend Dates in 2026])
                
                Flight Selected:
                - [Airline Name] (₹[Flight Price]) – Departs [Source City] at [Departure Time]
                
                Hotel Booked:
                - [Hotel Name] (₹[Price per night]/night, [Star Rating]-star)
                
                Weather:
                - Day 1: [Weather status forecast description] ([Max Temp]°C)
                - Day 2: [Weather status forecast description] ([Max Temp]°C)
                - Day 3: [Weather status forecast description] ([Max Temp]°C)
                
                Itinerary:
                Day 1: [List top 2 sightseeing attractions from places.json separated by commas]
                Day 2: [List next 2 sightseeing attractions from places.json separated by commas]
                Day 3: [List final 2 sightseeing attractions from places.json separated by commas]
                
                Estimated Total Budget:
                - Flight: ₹[Flight Price]
                - Hotel: ₹[Total Calculated Hotel Price for 2 nights]
                - Food & Travel: ₹[Estimated Local Expenses total]
                -------------------------------------
                Total Cost: ₹[Complete calculated total mathematical sum]
                """
                
                final_response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=final_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1
                    )
                )
                
                return {"output": final_response.text if final_response.text else "Itinerary generated successfully."}
                
            except Exception as e:
                # Catch 503 Overload and 429 Quota limits cleanly, wait 3 seconds, and retry loop
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    time.sleep(3)
                    continue
                # Gracefully fall back to an on-screen warning description instead of throwing a blank server crash
                return {"output": f"The AI server is very busy right now. Please click the button to try again in a moment! (Error details: {str(e)})"}

# Expose execution reference module instance
travel_agent_brain = AutonomousAgentExecutor()