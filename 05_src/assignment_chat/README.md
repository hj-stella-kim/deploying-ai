# AI system with conversational interface using custom services

## 1. Project Overview

This project involves the development of an intelligent conversational agent leveraging LangGraph to manage complex interaction flows. Rather than relying solely on pre-trained knowledge, the bot is designed to dynamically invoke specialized tools to retrieve real-time information. It functions as a context-aware assistant capable of handling specialized queries while maintaining a consistent and engaging persona.

## 2. The 3 Services I Built

### Service 1: API-Based Weather Lookup

* **Backend:** Uses a weather API which is AccuWeather (https://www.accuweather.com) to fetch real-time current weather data.
* **Transformation:** Instead of returning raw JSON data, the system summarizes the weather conditions into a natural, conversational sentence.

### Service 2: Semantic Music Review Search

* **Backend:** Implements lexical search using TF-IDF Vectorization (scikit-learn) on the provided class dataset of Pitchfork music reviews.
* **Functionality**: The review content is structured to prioritize Artist, Title, and text keywords. The user query is vectorized, and cosine similarity is calculated against the TF-IDF matrix of the entire dataset to find the most relevant reviews based on keyword matching. For queries requiring precise numerical data (e.g., "How many reviews have a score below 5?"), the system invokes the get_score_statistics tool (Tool Integration) to analyze the dataset and provide accurate counts, rather than relying on search result.

### Service 3: League of Legends Information Search (open-ended)

* **Backend:** Uses **Function Calling** and **Web Search** to trigger specific services based on user intent.
* **Functionality:** The service analyzes the query to determine if the user is looking for a schedule or general information of League of Legends. Based on this, it calls functions either WebSearch (for latest news) or EventSearch (for match schedules) using SerpAPI (https://serpapi.com/) to provide up-to-date information (Web Search)

## 3. User Interface (Gradio)

* **Framework:** Built with **Gradio Blocks** for a clean chat interface.
* **Personality:** The assistant acts as a knowledgeable, focused assistant helping users with their specific queries.
* **Memory:** The system maintains conversation history using **LangGraph's state management**, allowing for context-aware follow-up questions.

## 4. Guardrails and Security

To ensure safe and reliable interactions, the following guardrails are implemented:

* **Input Guardrails:** Filters out restricted keywords (Cats, Dogs, Horoscopes, Zodiac Signs, Taylor Swift) before they reach the LLM (1st validation). It also detects and blocks attempts to manipulate the system prompt (2nd).
* **Output Guardrails:** Monitors the AI's generated response. If a restricted topic is detected, the system automatically triggers a rewriting process to ensure the final output is compliant (3rd).

## 5. System Architecture

The application is structured into two main files:

1. **`main.py`**: Contains the LangGraph state definition, tool definitions, guardrail logic, and the agent compilation.
2. **`app.py`**: Manages the Gradio user interface and event handling.

To run the application, execute app.py. Here are some example prompts:
1. What is the current weather in {City Name}?
2. Summarize the League of Legends 14.5 patch notes. (Web Search - Google)
3. When and where will the 2026 League of Legends World Championship be held? (Web Search - Google Event)
4. Are there any songs with 'star' in the title? (Semantic Search)
5. How many Pitchfork music reviews have a score of 10? (Tool Call - Score Statistics)

I will share the API keys for the separate APIs I have written via Slack.
- AccuWeather ([https://www.accuweather.com](https://developer.accuweather.com/home)) : My free trial subscription for AccuWeather API expires on Mar 8, 2026. Please let me know if it needs to renewal.
- SerpAPI ([ttps://serpapi.com/](https://serpapi.com/)) 
