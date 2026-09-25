"""
Example 01: Fast ticket routing bootstrap with laya-ignite.
"""

import sys
from laya_ignite.schema import validate_question_bundle
from laya_ignite.synthesizer import create_generator, synthesize_dataset

questions = {
    "support_tier": {
        "type": "choice",
        "instructions": "Classify incoming tickets into the correct support tier",
        "criteria": {
            "tier1_faq": "Basic password resets, login troubleshooting, how-to guides",
            "tier2_billing": "Refund disputes, double charges, subscription cancellations",
            "tier3_bugs": "500 server crashes, database corruption, critical outages"
        }
    }
}

if __name__ == "__main__":
    print("[Laya-Ignite Example: Ticket Routing Bootstrap]")
    parsed = validate_question_bundle(questions)

    # Use mock generator for quick offline demonstration
    gen = create_generator("mock")
    dataset = synthesize_dataset(parsed, generator=gen, samples_per_class=5)

    print(f"Generated {len(dataset)} synthetic examples with adversarial hard negatives:")
    for i, ex in enumerate(dataset[:6], 1):
        tag = "[HARD NEGATIVE]" if ex.is_hard_negative else "[NORMAL]"
        print(f"{i}. {tag} Text: '{ex.text}' -> Target: {ex.target_labels}")
