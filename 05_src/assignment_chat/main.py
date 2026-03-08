import sys
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.tools import ToolException

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.append('../../05_src/')
load_dotenv('../../05_src/.secrets')

# custom services import
from assignment_chat.weather_service import get_weather_description
from assignment_chat.semantic_search_service import SemanticSearchService
from assignment_chat.lol_esport_service import classify_question, perform_web_search, perform_event_search
from assignment_chat.prompts import get_system_prompt

# ============================================================
# LLM Client Setup
# ============================================================
client = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
)

# ============================================================
# Music Service Instances
# ============================================================
music_service = None
def get_music_service():
    global music_service
    if music_service is None:
        print("[Loading Music Service...]")
        music_service = SemanticSearchService()
    return music_service

# ============================================================
# Tool Definitions
# ============================================================
@tool("SearchLeagueOfLegends", description="Search for League of Legends information, news, schedules, and tournament updates")
def search_lol(query: str) -> str:
    try:
        category = classify_question(query)
        if category == "lol_schedule":
            search_query = f"League of Legends schedule tournament event {query}"
            search_result = perform_event_search(search_query)
        else:
            search_query = f"League of Legends information {query}"
            search_result = perform_web_search(search_query)
        
        summary_prompt = f"""Based on the following League of Legends information, please provide a clear and organized answer to the user's question.
Question: {query}
Search Results: {search_result}
Provide a helpful and accurate answer based on the above results."""
        
        response = client.invoke([{"role": "user", "content": summary_prompt}])
        return response.content
    except Exception as e:
        raise ToolException(f"Error searching League of Legends: {str(e)}")

@tool("SearchMusicReviews", description="Search for music reviews, artist information, and music recommendations from Pitchfork database")
def search_music(query: str) -> str:
    try:
        service = get_music_service()
        search_results = service.search(query, top_k=10)
        answer = service.transform_response(search_results, query)
        return answer
    except Exception as e:
        raise ToolException(f"Error searching music reviews: {str(e)}")

@tool("GetWeather", description="Get current weather information for a specified city")
def get_weather(city: str) -> str:
    try:
        weather = get_weather_description(city)
        if weather is None:
            return f"Sorry, I couldn't find weather information for '{city}'. Please try another city name."
        return weather
    except Exception as e:
        raise ToolException(f"Error getting weather for {city}: {str(e)}")

tools = [search_lol, search_music, get_weather]

# ============================================================
# State Definition
# ============================================================
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ============================================================
# Guardrails
# ============================================================
restricted_keywords = ["cat", "dog", "horoscope", "zodiac", "taylor swift"]

def validate_input(user_input: str) -> str:
    """Input Guardrail: Checks user input before LLM processes it."""
    if any(keyword in user_input.lower() for keyword in restricted_keywords):
        return "I'm sorry, I cannot discuss restricted topics like pets, horoscopes, or certain celebrities."
    if "ignore previous instructions" in user_input.lower():
        return "Nice try, but I cannot ignore my safety instructions."
    return None

def output_guardrail_node(state: MessagesState):
    """Output Guardrail: Checks the final response for restricted content."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not isinstance(last_message, AIMessage):
        return {}

    response_content = last_message.content
    if any(keyword in response_content.lower() for keyword in restricted_keywords):
        print("[Guardrail] Restricted content detected. Rewriting response.")
        rewrite_prompt = "The previous response contained restricted topics (cats, dogs, horoscopes, or Taylor Swift). Please rewrite the response to exclude these topics politely."
        safe_response = client.invoke([SystemMessage(content=rewrite_prompt), *messages])
        return {"messages": [safe_response]}
    return {}

# ============================================================
# Graph Construction (Agent)
# ============================================================
def create_agent():
    """Create and return the agent executor graph with all tools."""
    
    # Bind tools to Client
    client_with_tools = client.bind_tools(tools)
    
    def chatbot(state: MessagesState):
        system_prompt = get_system_prompt()
        
        response = client_with_tools.invoke([
            {"role": "system", "content": system_prompt},
            *state["messages"]
        ])
        
        if isinstance(response, str):
            response = AIMessage(content=response)
        return {"messages": [response]}
    
    # Create the graph
    graph_builder = StateGraph(MessagesState)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_node("guardrail", output_guardrail_node)
    
    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    # Sequence 
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("chatbot", "guardrail")
    graph_builder.add_edge("guardrail", END)
    
    return graph_builder.compile()

# ============================================================
# Agent Interface
# ============================================================
agent = create_agent()


def chat(user_message: str, message_history: list) -> str:
    """Main chat function for Gradio interface."""

    # 1. Input Guardrail
    guardrail_response = validate_input(user_message)
    if guardrail_response:
        return guardrail_response
    
    # 2. History Conversion
    messages = []
    if message_history:
        for msg in message_history:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
    
    messages.append(HumanMessage(content=user_message))
    
    # 3. Invoke Agent
    result = agent.invoke({"messages": messages})
    
    # 4. Extract final response
    final_messages = result.get("messages", [])
    if final_messages:
        last_message = final_messages[-1]
        return last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    return "I encountered an error processing your request. Please try again."

if __name__ == "__main__":
    # Test the agent
    print("Testing agent...\n")
    test_queries = [
        "When is the 2026 League of Legends Worlds Championship?",
        "Find music reviews about jazz",
        "What's the weather in Seoul?"
    ]
    for query in test_queries:
        print(f"Q: {query}")
        response = chat(query, [])
        print(f"A: {response}\n")