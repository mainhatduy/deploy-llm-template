import httpx
import json
import sys

def test_endpoint():
    url = "http://localhost:8000/predict"
    
    # Example 1: Type 1 choice question
    payload_type1_choice = {
        "query_id": "T1_0001",
        "type": "type1",
        "query": "Is Student A eligible for graduation?",
        "premises": [
            "A student who has completed at least 120 credits is eligible for graduation.",
            "Student A has completed 118 credits."
        ],
        "options": ["Yes", "No", "Uncertain"]
    }
    
    # Example 2: Type 1 number/text question
    payload_type1_free = {
        "query_id": "T1_0002",
        "type": "type1",
        "query": "How many more credits does Student A need to graduate?",
        "premises": [
            "A student with >= 120 credits is eligible.",
            "Student A has 118 credits."
        ],
        "options": []
    }
    
    # Example 3: Type 2 physics problem
    payload_type2 = {
        "query_id": "T2_0001",
        "type": "type2",
        "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
        "premises": [],
        "options": []
    }
    
    payloads = [
        ("Type 1: Choice Question", payload_type1_choice),
        ("Type 1: Free-form Question", payload_type1_free),
        ("Type 2: Physics Problem", payload_type2)
    ]
    
    print("Sending requests to FastAPI Server /predict endpoint...\n")
    
    for label, payload in payloads:
        print(f"--- Sending {label} ---")
        print(json.dumps(payload, indent=2))
        try:
            response = httpx.post(url, json=payload, timeout=30.0)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Response Body:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error Response: {response.text}")
        except httpx.RequestError as e:
            print(f"Connection Error: {e}")
            print("Is the FastAPI server running?")
            sys.exit(1)
        print("-" * 40 + "\n")

if __name__ == "__main__":
    test_endpoint()
