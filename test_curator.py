#!/usr/bin/env python3
"""
Simple test to verify Curator works with LiteLLM + Bedrock setup
"""

from typing import Dict
from datasets import Dataset
from pydantic import BaseModel, Field
from bespokelabs import curator

class SimpleResponse(BaseModel):
    """Simple response format for testing."""
    response: str = Field(description="The response text")
    confidence: float = Field(description="Confidence score (0-1)")

class SimpleLLM(curator.LLM):
    """Simple LLM test class."""
    
    response_format = SimpleResponse
    
    def prompt(self, input: Dict) -> str:
        return f"Please respond to: {input['question']}"
    
    def parse(self, input: Dict, response: SimpleResponse) -> Dict:
        return {
            "question": input['question'],
            "answer": response.response,
            "confidence": response.confidence
        }

def main():
    print("🧪 Testing Curator with LiteLLM + Bedrock setup...")
    
    # Test data
    test_data = [
        {"question": "What is the capital of France?"},
        {"question": "Explain photosynthesis in one sentence."}
    ]
    
    dataset = Dataset.from_list(test_data)
    
    # Use Llama 3.3 70B model via LiteLLM proxy
    llm = SimpleLLM(
        model_name="llama-3-3-70b",
        backend="openai",
        backend_params={
            "base_url": "http://localhost:4000/v1",
            "api_key": "bedrock-graphgen-2024",
            "max_requests_per_minute": 10,
            "max_tokens_per_minute": 10000
        }
    )
    
    print("📊 Running test generation...")
    results = llm(dataset)
    
    print("✅ Test completed successfully!")
    
    # Use the dataset attribute to access results
    results_list = results.dataset.to_list()
    print(f"Generated {len(results_list)} responses")
    
    # Print results
    for i, result in enumerate(results_list):
        print(f"\n📋 Result {i+1}:")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")

if __name__ == "__main__":
    main()
