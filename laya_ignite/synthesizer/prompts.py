"""
Synthesis prompt templates for generating realistic training data and adversarial hard negatives.
"""

CHOICE_SYNTHESIS_PROMPT = """You are an expert AI data synthesizer. Your task is to generate {count} realistic, diverse text examples for a classification category.

Category Name: {label}
Category Description: {description}
Overall Task: {instructions}

Requirements:
1. Generate realistic, distinct inputs (e.g. support emails, messages, inquiries, or notes) that belong strictly to this category.
2. Vary the tone, vocabulary, length (some 1 sentence, some 2-3 sentences), and formatting across examples.
3. Do NOT include greetings like 'Dear Sir' or signatures unless relevant to the context.
4. Output strictly a JSON array of strings, without any extra conversation or markdown quotes:
["Example 1", "Example 2", ...]
"""

CHOICE_HARD_NEGATIVE_PROMPT = """You are an adversarial AI tester creating hard negative examples.
Target Category to AVOID: {label} ({description})
Alternative/Correct Category to TARGET: {other_label} ({other_description})
Overall Task: {instructions}

Generate {count} realistic examples that explicitly mention keywords related to '{label}', but clearly and unmistakably mean '{other_label}'.
Specifically use negations and contrastive phrasing such as:
- "I do NOT want {label}, please do {other_label}"
- "Unlike {label}, my issue is actually {other_label}"
- "I was thinking of {label}, but what I really need is {other_label}"

Output strictly a JSON array of strings:
["Example 1", "Example 2", ...]
"""

NOUL_SYNTHESIS_PROMPT = """You are an expert AI data synthesizer.
Task: {instructions}
Condition (True): {criteria}

Generate {count} realistic text examples where this statement is TRUE (positive examples), and {count} examples where this statement is FALSE (negative examples).

Output strictly a JSON object with two arrays:
{{"positive": ["Example 1", "Example 2", ...], "negative": ["Example 1", "Example 2", ...]}}
"""

SCORE_SYNTHESIS_PROMPT = """You are an expert AI data synthesizer.
Metric/Task: {instructions}
Scale Levels:
{levels_text}

Generate {count} distinct realistic text examples for Level {level_index} ({level_description}).
Output strictly a JSON array of strings:
["Example 1", "Example 2", ...]
"""
