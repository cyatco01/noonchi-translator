"""
Context-parsing evaluation.

Measures how accurately parse_situation() + FormalityResolver infer the correct
formality token from a plain-English situation description.

Run from backend/api/ with venv active:
  python ../evaluation/eval_context_parsing.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../api")

from anthropic import Anthropic
from agents.claude_agent import parse_situation, FormalityResolver
from models.schemas import FormalityToken

TEST_CASES = [
    # --- <formal> ---
    ("Emailing my professor about a missed deadline", FormalityToken.FORMAL),
    ("Presenting quarterly results to the board of directors", FormalityToken.FORMAL),
    ("Asking my boss for time off next week", FormalityToken.FORMAL),
    ("Meeting an elder relative I've never met before at a family gathering", FormalityToken.FORMAL),
    ("Ordering food from a waiter at a restaurant", FormalityToken.FORMAL),
    ("Introducing myself to a stranger at a networking event", FormalityToken.FORMAL),
    ("Asking a shop clerk where to find something", FormalityToken.FORMAL),
    # --- <polite> ---
    ("Catching up with a coworker I don't know well over lunch", FormalityToken.POLITE),
    ("Texting a classmate I'm not close with about the homework", FormalityToken.POLITE),
    ("Making small talk with my neighbor in the hallway", FormalityToken.POLITE),
    ("Asking a colleague to review my report before the deadline", FormalityToken.POLITE),
    ("Chatting with someone I just met at a party through mutual friends", FormalityToken.POLITE),
    ("Writing a message to a peer in a group project", FormalityToken.POLITE),
    # --- <casual> ---
    ("Texting my best friend to grab dinner tonight", FormalityToken.CASUAL),
    ("Talking to my younger sibling at home", FormalityToken.CASUAL),
    ("Playing video games with my childhood friends on a Friday night", FormalityToken.CASUAL),
    ("Joking around with my close college friends at someone's apartment", FormalityToken.CASUAL),
    # --- edge cases ---
    ("Chatting with a coworker who is also a close personal friend, at work", FormalityToken.POLITE),
    ("Talking to a friend who is 8 years older than me in a casual setting", FormalityToken.POLITE),
    ("Asking my direct report to finish a task, at the office", FormalityToken.POLITE),
]


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    resolver = FormalityResolver()

    passed = 0
    failures = []

    print(f"\nRunning {len(TEST_CASES)} context-parsing test cases...\n")
    print(f"{'#':<4} {'Expected':<10} {'Got':<10} {'Conf':<6} {'Pass':<5}  Situation")
    print("-" * 90)

    for i, (situation, expected) in enumerate(TEST_CASES, 1):
        try:
            context, reasoning, confidence = parse_situation(client, situation)
            predicted = resolver.resolve(context)
            ok = predicted == expected
            if ok:
                passed += 1
            else:
                failures.append((situation, expected, predicted, context, reasoning, confidence))

            conf_str = f"{confidence:.2f}"
            status = "PASS" if ok else "FAIL"
            short = situation[:55] + ("..." if len(situation) > 55 else "")
            print(f"{i:<4} {expected.value:<10} {predicted.value:<10} {conf_str:<6} {status:<5}  {short}")
        except Exception as e:
            failures.append((situation, expected, None, None, str(e), 0.0))
            print(f"{i:<4} {expected.value:<10} {'ERROR':<10} {'?':<6} {'FAIL':<5}  {situation[:55]}")

    total = len(TEST_CASES)
    accuracy = passed / total * 100
    print(f"\n{'='*90}")
    print(f"Accuracy: {passed}/{total} ({accuracy:.1f}%)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for situation, expected, predicted, context, reasoning, confidence in failures:
            print(f"\n  Situation : {situation}")
            print(f"  Expected  : {expected.value if expected else '?'}")
            print(f"  Predicted : {predicted.value if predicted else 'ERROR'}")
            if context:
                print(f"  Extracted : relationship={context.relationship.value}, "
                      f"age_diff={context.age_differential}, setting={context.setting.value}")
            print(f"  Reasoning : {reasoning}")


if __name__ == "__main__":
    main()
