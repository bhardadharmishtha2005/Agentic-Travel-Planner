import streamlit as st
import os
from agent import travel_agent_brain

# Configure the visual presentation of the page
st.set_page_config(page_title="Agentic AI Travel Planner", page_icon="✈️", layout="wide")

st.title("✈️ Agentic AI-Based Travel Planning Assistant")
st.markdown("##### Powered by LangChain & Streamlit")
st.write("Provide your travel details below, and our autonomous AI Agent will parse local datasets, fetch real-time weather forecasts, and construct an optimized custom itinerary for you.")

# Create columns for user input fields
col1, col2, col3 = st.columns(3)

with col1:
    source = st.text_input("🛫 Leaving From (Source City):", placeholder="e.g., Mumbai")
with col2:
    destination = st.text_input("🛬 Going To (Destination City):", placeholder="e.g., Goa")
with col3:
    max_budget = st.number_input("💰 Max Hotel Budget per Night ($/₹):", min_value=0, value=5000, step=500)

# Execution block
if st.button("Generate My Custom Itinerary ✨", use_container_width=True):
    if not source or not destination:
        st.warning("Please provide both a source and a destination city to begin.")
    else:
        with st.spinner(f"Agent is analyzing flights, hotels, and fetching weather updates for {destination}... Please wait."):
            try:
                # Constructing the instruction format for the LangChain executor
                user_query = f"Plan a trip from {source} to {destination}. My hotel budget per night is up to {max_budget}."
                
                # Executing the agent run
                response = travel_agent_brain.invoke({"input": user_query})
                
                # Rendering the final structured response from the LLM
                st.success("🎉 Your Itinerary is Ready!")
                st.markdown("### 🗺️ Custom Trip Plan")
                st.write(response.get("output", "No itinerary text returned."))
                
            except Exception as e:
                st.error(f"An error occurred during agent processing: {str(e)}")
                st.info("Tip: Please check that your GOOGLE_API_KEY is correctly saved in your .env file and your internet connection is active.")