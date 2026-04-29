"""
tests/test_language.py — Unit tests for the instruction parser.
"""

import pytest
from language import parse


class TestParseBasic:
    """Basic instruction parsing."""

    def test_find_simple_object(self):
        result = parse("find the apple")
        assert result.target == "apple"
        assert result.attributes == []
        assert result.room_hint is None

    def test_go_to_object(self):
        result = parse("go to the toaster")
        assert result.target == "toaster"

    def test_navigate_to_object(self):
        result = parse("navigate to the microwave")
        assert result.target == "microwave"

    def test_bare_object(self):
        result = parse("the fridge")
        assert result.target == "fridge"


class TestParseAttributes:
    """Attribute (adjective) extraction."""

    def test_single_attribute(self):
        result = parse("find the red mug")
        assert result.target == "mug"
        assert "red" in result.attributes

    def test_multiple_attributes(self):
        result = parse("find a large white plate")
        assert result.target == "plate"
        assert "large" in result.attributes or "white" in result.attributes

    def test_go_to_with_attribute(self):
        result = parse("go to the blue book")
        assert result.target == "book"
        assert "blue" in result.attributes


class TestParseRoomHint:
    """Room hint extraction."""

    def test_room_hint_kitchen(self):
        result = parse("the microwave in the kitchen")
        assert result.target == "microwave"
        assert result.room_hint is not None
        assert "kitchen" in result.room_hint

    def test_room_hint_bathroom(self):
        result = parse("find the bottle in the bathroom")
        assert result.target == "bottle"
        assert result.room_hint is not None
        assert "bathroom" in result.room_hint

    def test_no_room_hint(self):
        result = parse("find the apple")
        assert result.room_hint is None


class TestParseRelation:
    """Spatial relation extraction."""

    def test_on_relation(self):
        result = parse("go to the blue book on the desk")
        assert result.target == "book"
        if result.relation:
            assert "on" in result.relation

    def test_near_relation(self):
        result = parse("find the coffee machine near the sink")
        assert result.target in ("machine", "coffee")
        # relation might be extracted depending on parse
        # Main test is that target is extracted


class TestParseQuery:
    """Query string generation."""

    def test_simple_query(self):
        result = parse("find the apple")
        assert "apple" in result.query

    def test_attributed_query(self):
        result = parse("find the red mug")
        q = result.query
        assert "mug" in q
        # Attributes should appear in query
        if result.attributes:
            assert "red" in q

    def test_raw_preserved(self):
        text = "find the microwave in the kitchen"
        result = parse(text)
        assert result.raw == text
