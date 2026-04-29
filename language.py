"""
language.py — Natural Language Instruction Parser

Parses navigation instructions like "find the red mug on the kitchen counter"
into structured representations using spaCy dependency parsing + rules.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import spacy

# Load spaCy model (downloaded via: python -m spacy download en_core_web_sm)
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[language] Downloading spaCy model en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    _nlp = spacy.load("en_core_web_sm")


@dataclass
class ParsedInstruction:
    """Structured output from instruction parsing."""
    target: str                          # main object to find (e.g., "mug")
    attributes: List[str] = field(default_factory=list)  # adjectives (e.g., ["red"])
    room_hint: Optional[str] = None      # room context (e.g., "kitchen")
    relation: Optional[str] = None       # spatial relation (e.g., "on the counter")
    raw: str = ""                         # original instruction text

    @property
    def query(self) -> str:
        """Build a concise query string for the grounder."""
        parts = self.attributes + [self.target]
        q = " ".join(parts)
        if self.relation:
            q += f" {self.relation}"
        return q


# Room-related keywords for hint extraction
_ROOM_WORDS = {
    "kitchen", "bedroom", "bathroom", "living room", "livingroom",
    "dining room", "diningroom", "hallway", "garage", "office",
    "laundry", "pantry", "closet",
}

# Action verbs to strip from the beginning
_ACTION_VERBS = {
    "find", "go", "navigate", "move", "walk", "look", "search",
    "locate", "get", "fetch", "bring", "head", "proceed", "travel",
}


def parse(text: str) -> ParsedInstruction:
    """
    Parse a navigation instruction into a structured representation.

    Examples:
        "find the apple"            → target="apple"
        "go to the blue book"       → target="book", attributes=["blue"]
        "the microwave in the kitchen" → target="microwave", room_hint="kitchen"
        "find the red mug on the desk" → target="mug", attrs=["red"], relation="on the desk"

    Args:
        text: Natural language instruction.

    Returns:
        ParsedInstruction with extracted fields.
    """
    doc = _nlp(text.strip())

    target = None
    attributes = []
    room_hint = None
    relation = None

    # Strategy 1: Find the main noun via dependency parsing
    # Look for direct objects (dobj) or objects of prepositions (pobj)
    _NOUN_POS = {"NOUN", "PROPN"}
    candidate_nouns = []

    for token in doc:
        # Direct object of a verb
        if token.dep_ in ("dobj", "attr") and token.pos_ in _NOUN_POS:
            candidate_nouns.append(token)
        # Object of preposition (e.g., "go to the mug")
        elif token.dep_ == "pobj" and token.pos_ in _NOUN_POS:
            candidate_nouns.append(token)
        # Root noun (e.g., "the microwave in the kitchen" — no verb)
        elif token.dep_ == "ROOT" and token.pos_ in _NOUN_POS:
            candidate_nouns.append(token)

    # Strategy 2: If no dependency match, find the first non-room noun
    if not candidate_nouns:
        for token in doc:
            if token.pos_ in _NOUN_POS and token.text.lower() not in _ROOM_WORDS:
                candidate_nouns.append(token)

    # Strategy 3: If still nothing, try any NOUN/PROPN
    if not candidate_nouns:
        for token in doc:
            if token.pos_ in _NOUN_POS:
                candidate_nouns.append(token)

    # Pick the first candidate that isn't a room word as the target
    for noun in candidate_nouns:
        if noun.text.lower() not in _ROOM_WORDS:
            target = noun
            break

    # Fallback: use first candidate or last non-stop word
    if target is None and candidate_nouns:
        target = candidate_nouns[0]
    if target is None:
        # Last resort: use last non-stopword, non-verb token
        for token in reversed(doc):
            if not token.is_stop and token.pos_ not in ("VERB", "ADP", "PUNCT"):
                target = token
                break

    if target is None:
        # Absolute fallback: use the full text minus action verbs
        words = [w for w in text.lower().split()
                 if w not in _ACTION_VERBS and w not in ("the", "a", "an", "to")]
        return ParsedInstruction(
            target=words[-1] if words else text.strip(),
            raw=text,
        )

    # Extract adjectival modifiers of the target
    for child in target.children:
        if child.dep_ == "amod" and child.pos_ in ("ADJ", "PROPN", "NOUN"):
            attributes.append(child.text.lower())
    # Also check for compound nouns (e.g., "coffee machine")
    for child in target.children:
        if child.dep_ == "compound" and child.text.lower() not in attributes:
            attributes.append(child.text.lower())

    # Extract room hint: look for "in the <room>" pattern
    for token in doc:
        if token.text.lower() == "in" and token.dep_ == "prep":
            for child in token.children:
                if child.dep_ == "pobj":
                    # Check if it's a room word
                    room_text = child.text.lower()
                    # Also grab compound (e.g., "living room")
                    compounds = [c.text.lower() for c in child.children
                                 if c.dep_ in ("amod", "compound")]
                    if compounds:
                        room_text = " ".join(compounds + [room_text])
                    if any(rw in room_text for rw in _ROOM_WORDS):
                        room_hint = room_text
                        break

    # Extract spatial relation: "on the desk", "near the table", etc.
    relation_preps = {"on", "near", "next", "beside", "by", "under", "above", "behind"}
    for token in doc:
        if (token.text.lower() in relation_preps
                and token.dep_ == "prep"
                and token.head == target):
            # Build the full prepositional phrase
            rel_tokens = [token.text.lower()]
            for child in token.subtree:
                if child != token:
                    rel_tokens.append(child.text.lower())
            relation = " ".join(rel_tokens)
            break

    target_text = target.text.lower()

    return ParsedInstruction(
        target=target_text,
        attributes=attributes,
        room_hint=room_hint,
        relation=relation,
        raw=text,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_instructions = [
        "find the apple",
        "go to the blue book on the desk",
        "the microwave in the kitchen",
        "navigate to a red mug",
        "find the coffee machine near the sink",
        "look for the green bottle in the bathroom",
        "go to the toaster",
        "find a large white plate on the counter",
    ]

    for instr in test_instructions:
        result = parse(instr)
        print(f"  Input:  {instr}")
        print(f"  Target: {result.target}, Attrs: {result.attributes}, "
              f"Room: {result.room_hint}, Rel: {result.relation}")
        print(f"  Query:  {result.query}")
        print()
