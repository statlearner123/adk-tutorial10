"""
Comprehensive pytest test suite for support_agent — Tutorial 10.

Test class breakdown (22 total):
    TestToolFunctions       (10) — individual tool behaviour & error handling
    TestAgentConfiguration   (7) — agent init, name, tools, model, metadata
    TestIntegration          (2) — multi-step orchestration workflows
    TestAgentEvaluation      (3) — async AgentEvaluator trajectory + response

Run all:
    pytest support_agent_tests/ -v

Run only fast unit tests (no API):
    pytest support_agent_tests/ -k "not TestAgentEvaluation" -v

Run only evaluation tests:
    pytest support_agent_tests/test_agent.py::TestAgentEvaluation -v
"""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

from support_agent.agent import (
    MODEL,
    check_ticket_status,
    create_ticket,
    root_agent,
    search_knowledge_base,
)

TESTS_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Tool Functions  (10 tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestToolFunctions:
    """Test each tool function in isolation using mock ToolContext."""

    def setup_method(self):
        self.ctx = Mock()
        self.ctx.tickets = {}

    # search_knowledge_base ───────────────────────────────────────────────────

    def test_search_kb_password_reset(self):
        """KB search finds password-reset article."""
        result = search_knowledge_base("password reset", self.ctx)
        assert result["status"] == "success"
        assert "password" in result["report"].lower()
        assert len(result["results"]) > 0
        assert "Reset Password" in result["results"][0]["content"]

    def test_search_kb_refund_policy(self):
        """KB search finds refund-policy article."""
        result = search_knowledge_base("refund", self.ctx)
        assert result["status"] == "success"
        assert len(result["results"]) > 0
        assert "30-day" in result["results"][0]["content"]

    def test_search_kb_shipping(self):
        """KB search finds shipping-info article."""
        result = search_knowledge_base("shipping", self.ctx)
        assert result["status"] == "success"
        assert len(result["results"]) > 0
        assert "3-5 business days" in result["results"][0]["content"]

    def test_search_kb_not_found(self):
        """KB search returns empty results for unknown topic."""
        result = search_knowledge_base("nonexistent topic xyz", self.ctx)
        assert result["status"] == "success"
        assert "no articles found" in result["report"].lower()
        assert result["results"] == []

    # create_ticket ───────────────────────────────────────────────────────────

    def test_create_ticket_medium_priority(self):
        """Ticket creation with medium priority."""
        result = create_ticket("My account is locked", self.ctx, "medium")
        assert result["status"] == "success"
        assert "created successfully" in result["report"]
        assert "medium" in result["report"]
        assert result["ticket"]["priority"] == "medium"
        assert result["ticket"]["status"] == "open"
        assert "ticket_id" in result["ticket"]

    def test_create_ticket_high_priority(self):
        """Ticket creation with high priority gets 24h response window."""
        result = create_ticket("Website is down", self.ctx, "high")
        assert result["status"] == "success"
        assert "high" in result["report"]
        assert result["ticket"]["priority"] == "high"
        assert result["ticket"]["estimated_response"] == "24 hours"

    def test_create_ticket_invalid_priority(self):
        """Invalid priority returns error without ticket key."""
        result = create_ticket("Test issue", self.ctx, "invalid")
        assert result["status"] == "error"
        assert "Invalid priority" in result["error"]
        assert "ticket" not in result

    def test_create_ticket_unique_ids(self):
        """Different issues produce different ticket IDs."""
        r1 = create_ticket("Issue 1", self.ctx)
        r2 = create_ticket("Issue 2", self.ctx)
        assert r1["ticket"]["ticket_id"] != r2["ticket"]["ticket_id"]

    # check_ticket_status ─────────────────────────────────────────────────────

    def test_check_ticket_status_existing(self):
        """Ticket created in same context is findable by check_ticket_status."""
        create_result = create_ticket("Test issue", self.ctx)
        ticket_id = create_result["ticket"]["ticket_id"]

        status_result = check_ticket_status(ticket_id, self.ctx)
        assert status_result["status"] == "success"
        assert ticket_id in status_result["report"]
        assert status_result["ticket"]["status"] == "open"

    def test_check_ticket_status_not_found(self):
        """Non-existent ticket returns error without ticket key."""
        result = check_ticket_status("TICK-NONEXISTENT", self.ctx)
        assert result["status"] == "error"
        assert "not found" in result["error"]
        assert "ticket" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Agent Configuration  (7 tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentConfiguration:
    """Validate the root_agent object's setup."""

    def test_agent_exists(self):
        assert root_agent is not None
        assert hasattr(root_agent, "name")

    def test_agent_name(self):
        assert root_agent.name == "support_agent"

    def test_agent_has_tools(self):
        tool_names = [t.__name__ for t in root_agent.tools]
        assert "search_knowledge_base" in tool_names
        assert "create_ticket" in tool_names
        assert "check_ticket_status" in tool_names

    def test_agent_model(self):
        assert root_agent.model == MODEL

    def test_agent_has_description(self):
        assert root_agent.description is not None
        assert "support" in root_agent.description.lower()

    def test_agent_has_instruction(self):
        assert root_agent.instruction is not None
        assert len(root_agent.instruction) > 0

    def test_agent_output_key(self):
        assert root_agent.output_key == "support_response"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Integration Workflows  (2 tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    """End-to-end multi-step tool orchestration (no API calls)."""

    def setup_method(self):
        self.ctx = Mock()
        self.ctx.tickets = {}

    def test_knowledge_base_completeness(self):
        """KB covers all expected topic keywords."""
        topics = ["password", "refund", "shipping", "account", "billing", "technical"]
        for topic in topics:
            result = search_knowledge_base(topic, self.ctx)
            assert result["status"] == "success"
            assert len(result["results"]) > 0, f"No KB results for topic: {topic!r}"

    def test_ticket_creation_workflow(self):
        """Create a ticket, then check its status — full round-trip."""
        create_result = create_ticket("Website loading slowly", self.ctx, "high")
        assert create_result["status"] == "success"

        ticket_id = create_result["ticket"]["ticket_id"]
        status_result = check_ticket_status(ticket_id, self.ctx)

        assert status_result["status"] == "success"
        assert status_result["ticket"]["ticket_id"] == ticket_id
        assert status_result["ticket"]["status"] == "open"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Agent Evaluation  (3 tests — requires Gemini API)
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentEvaluation:
    """
    Trajectory + response-quality evaluation via AgentEvaluator.
    These tests make *real* Gemini API calls — keep num_runs=1 to avoid
    rate-limit errors.
    """

    @pytest.mark.asyncio
    async def test_simple_kb_search(self):
        """Eval: agent searches KB and returns password-reset answer."""
        await AgentEvaluator.evaluate(
            agent_module="support_agent",
            eval_dataset_file_path_or_dir=str(TESTS_DIR / "simple.test.json"),
            num_runs=1,
        )

    @pytest.mark.asyncio
    async def test_ticket_creation(self):
        """Eval: agent searches KB then creates a ticket."""
        await AgentEvaluator.evaluate(
            agent_module="support_agent",
            eval_dataset_file_path_or_dir=str(
                TESTS_DIR / "ticket_creation.test.json"
            ),
            num_runs=1,
        )

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Eval: agent handles a two-turn conversation with context."""
        await AgentEvaluator.evaluate(
            agent_module="support_agent",
            eval_dataset_file_path_or_dir=str(TESTS_DIR / "complex.evalset.json"),
            num_runs=1,
        )
