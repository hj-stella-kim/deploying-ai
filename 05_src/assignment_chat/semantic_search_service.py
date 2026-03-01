import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

class SemanticSearchService: 
    def __init__(self):
        """
        Initialize the service: Load data, merge datasets, and build the search index.
        """
        self.model = self.load_model() 
        
        # Load datasets from JSONL files
        reviews = self.load_pitchfork_data("pitchfork_reviews.jsonl")
        content = self.load_pitchfork_data("pitchfork_content.jsonl")
        
        # Merge review metadata with long-form review content
        self.records = self.merge_datasets(reviews, content)
        
        # Initialize TF-IDF Vectorizer for Lexical/Hybrid search
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Use the 'text_format' for indexing instead of raw JSON
        # This ensures the search engine prioritizes Artist, Title, and Content keywords.
        texts = [r.get('text_format', 'empty') for r in self.records]
        
        if not texts:
            texts = ['empty']
            
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
      
    def load_model(self):
        """
        Setup OpenAI client using environment variables and custom gateway.
        """
        sys.path.append('../../05_src/')
        load_dotenv('../../05_src/.secrets')

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set.")
            return None
        
        return OpenAI(
            base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
            api_key=api_key,
            default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
        )
        
    def load_pitchfork_data(self, filename: Optional[str] = None) -> List[Dict]:
        """Load JSONL files safely."""
        path = Path(__file__).resolve().parents[1] / "documents" / filename

        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        data = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError: continue
        return data

    def merge_datasets(self, reviews: List[Dict], content: List[Dict]) -> List[Dict]:
        """
        Merges two datasets and creates a 'text_format' field optimized for LLM and Search.
        """
        content_map = {item.get('reviewid'): item for item in content}
        merged = []
        for review in reviews:
            review_id = review.get('reviewid')
            if review_id in content_map:
                merged_record = {**review, **content_map[review_id]}
                
                # Structured string for better keyword matching and LLM comprehension
                text_format = (
                    f"Title: {merged_record.get('title', '')}. "
                    f"Artist: {merged_record.get('artist', 'Unknown')}. "
                    f"Author: {merged_record.get('author', 'Unknown')}. " 
                    f"Score: {merged_record.get('score', 0)}. "
                    f"Published on: {merged_record.get('pub_date', 'N/A')}. "
                    f"Review Content: {merged_record.get('content', '')}"
                )
                merged_record['text_format'] = text_format
                merged.append(merged_record)
        
        print(f"Successfully merged {len(merged)} records.")
        return merged

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Calculates similarity between the query and the entire dataset.
        """
        print("perform music review search from Pitchfork music reviews database")
        # 1. Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([query])
        
        # 2. Compare query_vec WITH self.tfidf_matrix (the entire database)
        # Without self.tfidf_matrix, it just compares the query to itself.
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 3. Sort by similarity score in descending order
        best_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in best_indices:
            score = float(similarities[idx])
            if score > 0.02:  # exclude data having score almost zero
                results.append({
                    "record": self.records[idx],
                    "score": score
                })
        return results
    
    def get_score_statistics(self, threshold: float, comparison: str = "under"):
        """
        Scans the entire dataset and returns both the count AND representative samples.
        """
        matched_records = []
        if comparison == "under":
            matched_records = [r for r in self.records if r.get('score', -1) <= threshold]
        elif comparison == "over":
            matched_records = [r for r in self.records if r.get('score', -1) >= threshold]
        elif comparison == "equals" or comparison == "is":
            matched_records = [r for r in self.records if r.get('score', -1) == threshold]
        
        count = len(matched_records)
        
        # return three examples 
        samples = []
        for r in matched_records[:3]:
            samples.append({
                "artist": r.get("artist"),
                "title": r.get("title"),
                "score": r.get("score"),
                "pub_date": r.get("pub_date")
            })
        
        return {
            "total_count": count, 
            "threshold": threshold, 
            "comparison": comparison,
            "example_samples": samples
        }

    def transform_response(self, search_results: List[Dict], user_query: str) -> str:
        """
        Synthesizes search results and tool outputs into a natural language response.
        Handles Relevance Check, Summary, and Sentiment Analysis.
        """
        if not search_results or search_results[0]['score'] < 0.01:
            return "I'm sorry, I couldn't find any relevant music reviews matching your request."

        # Prepare context for the LLM
        context_data = "\n---\n".join([json.dumps(res['record'], ensure_ascii=False) for res in search_results])

        # Define tool for OpenAI Function Calling
        tools = [{
            "type": "function",
            "function": {
                "name": "get_score_statistics",
                "description": "Calculate review counts from the FULL database based on score thresholds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "threshold": {"type": "number"},
                        "comparison": {"type": "string", "enum": ["under", "over", "equals", "is"]}
                    },
                    "required": ["threshold", "comparison"]
                }
            }
        }]

        system_prompt = (
            "You are a highly capable Music review Agent. Your goal is to provide precise information using both search results and specialized tools.\n\n"
            "OPERATIONAL RULES:\n"
            "1. SPECIFIC QUERIES: When the user asks about a specific Artist or Title, find exact matches in the [Retrieved Data Samples].\n"
            "2. NUMERICAL QUERIES: For questions regarding counts, averages, or score thresholds about score (e.g., 'how many', 'more than X'), you MUST use the 'get_score_statistics' tool for 100% accuracy.\n"
            "3. EXAMPLE REQUESTS: If a user asks for examples (e.g., 'Give me an example') after a statistical query, cross-reference the tool output with the [Retrieved Data Samples].\n"
            "4. NO HALLUCINATION: If you found a total count via the tool but the specific example is missing from the samples, clearly state: 'I found X records in the database, but the specific details for those are not in my current top search results. Here is the most relevant one I found instead:' then provide the best match available."
            "5. SCORE DISTINCTION: You must distinguish between 'Similarity Score' (search relevance) and 'Review Score' (the album's rating).\n"
            "6. ACCURACY GUARDRAIL: A 'Review Score' of 0.0 is a failing grade. A 'Review Score' of 10.0 is perfect. Never confuse these two values. If the search results do not contain an exact 10.0 score, do not claim to find one."
            ""
        )

        user_prompt = f"""
            [User Question]: "{user_query}"
            
            [Retrieved Data Samples (Use these to answer)]: 
            {context_data}
            """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # First LLM Call: Reasoning and Tool Selection
        response = self.model.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If Tool is needed (e.g., "How many reviews are below 5?")
        if tool_calls:
            messages.append(response_message) 

            for tool_call in tool_calls:
                args = json.loads(tool_call.function.arguments)
                tool_output = self.get_score_statistics(
                    threshold=args.get("threshold"),
                    comparison=args.get("comparison", "under")
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "get_score_statistics",
                    "content": json.dumps(tool_output) 
                })
            

            final_response = self.model.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return final_response.choices[0].message.content

        return response_message.content

# Testing
if __name__ == "__main__":
    print("Initializing Music Discovery Service... Please wait.")
    service = SemanticSearchService()
    print("Service Ready! (Type 'Q' or 'quit' to exit)\n")

    while True:
        user_q = input("User: ").strip()

        if user_q.upper() in ["Q", "QUIT", "EXIT"]:
            print("Goodbye! Happy listening.")
            break

        if not user_q:
            continue

        # Using top_k=10 to provide enough context for the LLM to summarize
        search_res = service.search(user_q, 10)
        
        print("\nSearching and analyzing...")
        final_answer = service.transform_response(search_res, user_q)
        
        print(f"\n--- AI Response ---\n{final_answer}\n")
        print("-" * 50)