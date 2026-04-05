from src.explanation import generate_llm_explanation

# Simulate the local LLM initialization safely
try:
    from transformers import pipeline
    print("Loading google/flan-t5-small...")
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    generator = None

# Test Suite: Various JSON structural bindings
tests = [
    {
        "stage": "Severe",
        "json": {"intensity": "high", "spread": "diffuse", "location": "macula", "area_percent": 35.5}
    },
    {
        "stage": "Moderate",
        "json": {"intensity": "medium", "spread": "localized", "location": "peripheral", "area_percent": 12.0}
    },
    {
        "stage": "Mild",
        "json": {"intensity": "low", "spread": "localized", "location": "mixed", "area_percent": 3.2}
    }
]

for i, test in enumerate(tests):
    print(f"\\n--- RUNNING SCENARIO {i+1} ---")
    output = generate_llm_explanation(test["json"], test["stage"], generator)
    print(f"FINAL OUTPUT: {output}")

