"""
Test script for the two-step translation flow.

Usage:
    1. Start the server: python app.py
    2. Run this script: python test_api.py
"""

import requests
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_json(data: Dict[Any, Any]):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def test_health():
    print_section("Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print_json(response.json())
    assert response.status_code == 200
    print("✓ Health check passed")


def test_two_step_flow():
    """Test the complete two-step flow with a boss/workplace context."""
    print_section("STEP 1: Set Context (boss, workplace, speaker younger)")

    context_request = {
        "relationship": "boss",
        "age_differential": -10,
        "setting": "workplace"
    }
    print("Request:")
    print_json(context_request)

    response = requests.post(f"{BASE_URL}/api/set-context", json=context_request)
    print(f"\nStatus: {response.status_code}")
    context_response = response.json()
    print_json(context_response)

    assert response.status_code == 200
    session_id = context_response["session_id"]
    print(f"\n✓ Context set → {context_response['conditioning_input_prefix']}")

    print_section("STEP 2: Translate")
    translation_request = {"session_id": session_id, "text": "Do you want to eat?"}
    print("Request:")
    print_json(translation_request)

    response = requests.post(f"{BASE_URL}/api/translate", json=translation_request)
    print(f"\nStatus: {response.status_code}")
    result = response.json()
    print_json(result)

    assert response.status_code == 200
    print(f"\n✓ Conditioned input: {result['conditioned_input']}")
    print(f"  Korean output:     {result['translated_text']}")

    return session_id


def test_multiple_translations_same_session(session_id: str):
    print_section("Multiple Translations — Same Session")

    phrases = [
        "Would you like some coffee?",
        "Let's discuss the quarterly results.",
        "Thank you for your time."
    ]

    for i, phrase in enumerate(phrases, 1):
        print(f"\n[{i}] {phrase}")
        response = requests.post(
            f"{BASE_URL}/api/translate",
            json={"session_id": session_id, "text": phrase}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"    Conditioned: {result['conditioned_input']}")
            print(f"    Korean:      {result['translated_text']}")
        else:
            print(f"    Error {response.status_code}: {response.json()}")


def test_formality_levels():
    """Show the same phrase across all three formality levels."""
    print_section("Formality Comparison — Same Phrase, Three Contexts")

    test_cases = [
        {"relationship": "boss",     "age_differential": -10, "setting": "workplace",
         "label": "FORMAL  (boss, workplace, speaker younger)"},
        {"relationship": "colleague", "age_differential": 0,   "setting": "workplace",
         "label": "POLITE  (colleague, workplace, same age)"},
        {"relationship": "friend",    "age_differential": 0,   "setting": "intimate",
         "label": "CASUAL  (friend, intimate, same age)"},
    ]

    phrase = "Do you want to eat?"

    for case in test_cases:
        label = case.pop("label")
        print(f"\n--- {label} ---")

        ctx_response = requests.post(f"{BASE_URL}/api/set-context", json=case).json()
        session_id = ctx_response["session_id"]
        token = ctx_response["conditioning_input_prefix"]
        print(f"Token: {token}")

        tr_response = requests.post(
            f"{BASE_URL}/api/translate",
            json={"session_id": session_id, "text": phrase}
        ).json()

        print(f"Input:  {tr_response['conditioned_input']}")
        print(f"Output: {tr_response['translated_text']}")
        if tr_response.get("romanization"):
            print(f"Roman:  {tr_response['romanization']}")


def test_formality_override():
    """Test that formality_override takes precedence over inferred token."""
    print_section("Formality Override")

    context_request = {
        "relationship": "boss",
        "age_differential": -10,
        "setting": "workplace",
        "formality_override": "casual"
    }
    print("Context (boss/workplace but override=casual):")
    print_json(context_request)

    ctx_response = requests.post(f"{BASE_URL}/api/set-context", json=context_request).json()
    print(f"\nResolved token: {ctx_response['conditioning_input_prefix']}")
    assert ctx_response["formality_token"] == "casual", "Override should have taken effect"
    print("✓ Override correctly applied")


def test_invalid_session():
    print_section("Error Handling — Invalid Session")

    response = requests.post(
        f"{BASE_URL}/api/translate",
        json={"session_id": "invalid-id", "text": "Hello"}
    )
    print(f"Status: {response.status_code}")
    print_json(response.json())
    assert response.status_code == 404
    print("✓ Correctly returned 404")


def main():
    print("\n" + "=" * 60)
    print("  Noonchi Translator — API Test Suite")
    print("=" * 60)

    try:
        test_health()
        session_id = test_two_step_flow()
        test_multiple_translations_same_session(session_id)
        test_formality_levels()
        test_formality_override()
        test_invalid_session()

        print_section("All Tests Passed")
        print("\nFlow: social context → FormalityResolver → <token> English → Korean")

    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect. Start the server first:")
        print("   python app.py")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
