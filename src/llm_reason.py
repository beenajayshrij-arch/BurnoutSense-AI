"""
Uses Groq's free API (Llama 3.3 70B) to generate a short, professional
sentence for an HR dashboard explaining an employee's burnout risk.
A pure leaf node: it is NEVER parsed, NEVER used to influence action_map or
any other decision logic. Only SHAP output and rule-based logic
(action_mapping.py) drive decisions. This function only produces
human-readable text.

Uses model: llama-3.3-70b-versatile (Groq free tier).
Reads the API key from the environment variable GROQ_API_KEY.
"""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You write exactly one short, professional sentence for an HR dashboard. "
    "Use ONLY the facts provided. Do not invent causes, numbers, or actions."
)


def generate_reason(risk_level, top_feature, second_feature):
    api_key = os.environ.get("GROQ_API_KEY")
    user_prompt = (
        f"Employee burnout risk level: {risk_level}. "
        f"Top contributing factors: {top_feature} and {second_feature}. "
        f"Write one sentence summarizing this for an HR dashboard."
    )

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 100,
                "temperature": 0.3,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from Groq")

        reason_text = choices[0].get("message", {}).get("content", "").strip()
        if not reason_text:
            raise ValueError("Empty response from LLM")

        return reason_text

    except Exception as e:
        print(f"[generate_reason] LLM call failed ({e}), using fallback template.")
        return _fallback_reason(risk_level, top_feature, second_feature)


def _fallback_reason(risk_level, top_feature, second_feature):
    """Deterministic fallback sentence used if the LLM call fails for any reason."""
    return f"This employee shows {risk_level} burnout risk, primarily driven by {top_feature} and {second_feature}."


if __name__ == "__main__":
    print("[llm_reason] Testing generate_reason() with a sample employee:")
    reason = generate_reason("High", "Mental Fatigue Score", "Resource Allocation")
    print(f"  Result: {reason}")