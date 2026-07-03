import json
import os
import sys
from dotenv import load_dotenv



load_dotenv()

# Add the parent folder to the path so we can import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector import retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Set up the model and chain — same as main.py
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
model = ChatOpenAI(model=MODEL_NAME, api_key=os.getenv("OPENAI_API_KEY"))

template = """
You are an expert in answering questions about a pizza restaurant based on customer reviews.
Use only the reviews provided below to answer the question. Do not make up information that 
isn't supported by the reviews. If the reviews don't contain enough information to answer 
the question, say so.

Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# --------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------

def load_event(filename: str) -> dict:
    """Load a test question from a JSON file"""
    filepath = os.path.join(os.path.dirname(__file__), "events", filename)
    with open(filepath, "r") as f:
        return json.load(f)

def ask_chatbot(question: str) -> str:
    """Ask the chatbot a question and get the answer as a string"""
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    return result.content

# --------------------------------------------------------------
# Test functions
# --------------------------------------------------------------

def test_pizza_quality_question():
    """Test that the chatbot gives a real answer about pizza quality"""
    event = load_event("pizza_quality.json")
    answer = ask_chatbot(event["question"])

    # Check 1 — did it actually give an answer?
    assert len(answer) > 50, "Answer is too short — probably didn't find relevant reviews"

    # Check 2 — did it mention pizza related words?
    answer_lower = answer.lower()
    assert any(word in answer_lower for word in ["pizza", "crust", "sauce", "cheese", "topping"]), \
        "Answer doesn't mention any pizza-related words"

    # Check 3 — did it not just say it doesn't know?
    assert "i don't know" not in answer_lower, \
        "Chatbot said it doesn't know — but it should find pizza reviews"

def test_mozzarella_question():
    """Test that the chatbot finds mozzarella information from reviews"""
    event = load_event("mozzarella.json")
    answer = ask_chatbot(event["question"])

    # Check 1 — answer exists
    assert len(answer) > 20, "Answer too short"

    # Check 2 — mozzarella actually mentioned
    assert "mozzarella" in answer.lower(), \
        "Answer doesn't mention mozzarella even though the question asks about it"

def test_negative_review_question():
    """Test that the chatbot can surface negative feedback"""
    event = load_event("negative_review.json")
    answer = ask_chatbot(event["question"])

    # Check 1 — answer exists
    assert len(answer) > 50, "Answer too short"

    # Check 2 — answer mentions some kind of criticism or complaint
    answer_lower = answer.lower()
    negative_words = ["complaint", "issue", "problem", "disappoint", "bad", "poor", 
                      "overpriced", "slow", "rude", "negative", "concern"]
    assert any(word in answer_lower for word in negative_words), \
        "Answer doesn't mention any negative feedback even though reviews contain complaints"

def test_answer_stays_grounded():
    answer = ask_chatbot("What is the capital of France?")
    answer_lower = answer.lower()

    assert any(phrase in answer_lower for phrase in [
        "don't know",
        "not enough information",
        "not mentioned",
        "reviews don't",
        "cannot find",
        "only answer questions",
        "not related",
        "do not contain",
        "cannot answer",
        "no information",
    ]), "Chatbot answered a geography question it shouldn't know about"

# --------------------------------------------------------------
# Run all tests
# --------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_pizza_quality_question,
        test_mozzarella_question,
        test_negative_review_question,
        test_answer_stays_grounded,
    ]

    passed = 0
    failed = 0

    print("Running chatbot evaluations...\n")

    for test in tests:
        try:
            test()
            print(f"{test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"{test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"{test.__name__} — unexpected error: {e}")
            failed += 1

    print(f"\nResults: {passed}/{len(tests)} tests passed")

    if failed > 0:
        print("\nSome tests failed. Review the errors above.")
    else:
        print("\nAll tests passed!")