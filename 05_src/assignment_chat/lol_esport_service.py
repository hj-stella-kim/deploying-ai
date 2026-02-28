import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import tool


sys.path.append('../../05_src/')
load_dotenv('../../05_src/.secrets')



# ============================================================
# Internal search functions
# ============================================================
def _web_search_impl(query: str) -> str:
    """Google search implementation"""
    google_search = SerpAPIWrapper(
        search_engine="google",
        serpapi_api_key=os.getenv("SERP_API_KEY"),
    )
    return google_search.run(query)


def _event_search_impl(query: str) -> str:
    """Google Events search implementation"""
    google_search = SerpAPIWrapper(
        search_engine="google_events",
        serpapi_api_key=os.getenv("SERP_API_KEY"),
    )
    return google_search.run(query)


# Tool decorators (optional - use only when tool binding is needed)
@tool("WebSearch", response_format="content", description="Use this tool when searching the latest news and information about League of Legends.")
def web_search(query: str):
    return _web_search_impl(query)


@tool("EventSearch", response_format='content', description="Use this tool when searching events and schedules about League of Legends.")
def event_search(query: str):
    return _event_search_impl(query)

client = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')})

# ============================================================
# Question classification function
# ============================================================
def classify_question(question):
    """
    Classify user question into categories:
    - "not_lol": Questions unrelated to League of Legends
    - "lol_general": General League of Legends questions
    - "lol_news": League of Legends latest news/information questions
    - "lol_schedule": League of Legends schedule/match information questions
    """
    classification_prompt = f"""You are an expert League of Legends (LoL) specialist bot.
Please accurately classify the user's question into one of the following categories.

User Question: "{question}"

Classification Criteria:
- not_lol: Questions unrelated to League of Legends (e.g., electric vehicles, weather)
- lol_general: General League of Legends information (gameplay, champions, meta, etc.) - no need for latest info
- lol_news: Latest League of Legends news, updates, issues - requires up-to-date information
- lol_schedule: League of Legends match schedules, tournament schedules, match results - requires schedule/event information

Response Format: Write exactly one of the following:
not_lol
lol_general
lol_news
lol_schedule"""
    
    classification_response = client.invoke(classification_prompt)
    category = classification_response.content.strip().lower()
    return category

# ============================================================
# Main logic
# ============================================================
question = "When is the 2026 League of Legends Worlds Championship?"

print(f"Question: {question}\n")

# Step 1: Classify the question
category = classify_question(question)
print(f"[Classification Result] {category}\n")

# Step 2: Process based on classification
if category == "not_lol":
    print("❌ Sorry, this service only handles League of Legends-related questions.")
    
elif category == "lol_general":
    # Use base LLM (no search, answer with general knowledge)
    print("[Processing] Using base LLM to provide answer.\n")
    response = client.invoke(question)
    print(f"📝 Answer:\n{response.content}")
    
elif category == "lol_news":
    # Use web search (search for latest news)
    print("[Processing] Using web search to find latest news.\n")
    search_query = f"League of Legends latest news information {question}"
    search_result = _web_search_impl(search_query)
    
    # Organize search results and provide answer
    summary_prompt = f"""Based on the following League of Legends information, please answer the user's question.
Question: {question}

Search Results:
{search_result}

Provide a clear and well-organized answer based on the above results."""
    
    summary_response = client.invoke(summary_prompt)
    print(f"📝 Answer:\n{summary_response.content}")
    
elif category == "lol_schedule":
    # Use event search (search for match schedules)
    print("[Processing] Searching for match schedules.\n")
    search_query = f"League of Legends schedule tournament event {question}"
    event_result = _event_search_impl(search_query)
    
    # Organize schedule information and provide answer
    summary_prompt = f"""Based on the following League of Legends match schedule information, please answer the user's question.
Question: {question}

Match Schedule Information:
{event_result}

Provide a clear and well-organized answer based on the above information."""
    
    summary_response = client.invoke(summary_prompt)
    print(f"📝 Answer:\n{summary_response.content}")
    
else:
    print(f"⚠️  Unknown classification result: {category}")