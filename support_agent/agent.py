"""
Customer Support Agent — For Evaluation & Testing Demonstration (Tutorial 10)

Uses Gemini API directly (GOOGLE_GENAI_USE_VERTEXAI=FALSE).

Tools:
    search_knowledge_base   — search KB articles
    create_ticket           — open a support ticket
    check_ticket_status     — look up a ticket
"""

import hashlib
from typing import Any, Dict

from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext

MODEL = "gemini-2.5-flash"

# ── Ticket Storage Helpers ───────────────────────────────────────────────────

def _get_ticket_store(tool_context: ToolContext) -> Dict[str, Any]:
    """Return a mutable dict that persists ticket data on tool_context."""
    # Works with real ADK (state dict) and unittest.mock.Mock (tickets attr)
    try:
        if hasattr(tool_context, "state") and isinstance(
            getattr(tool_context, "state", None), dict
        ):
            store = tool_context.state
            if "_tickets" not in store:
                store["_tickets"] = {}
            return store["_tickets"]
    except Exception:
        pass
    # Fallback for Mock-based unit tests
    if not hasattr(tool_context, "tickets") or not isinstance(
        tool_context.tickets, dict
    ):
        tool_context.tickets = {}
    return tool_context.tickets


# ── Tools ────────────────────────────────────────────────────────────────────

def search_knowledge_base(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Search the knowledge base for articles matching the query.

    Args:
        query: Search query string.
    """
    kb = {
        "password reset": (
            "To reset your password, go to Settings > Security > Reset Password."
        ),
        "refund policy": (
            "30-day money-back guarantee. Contact support@example.com for refunds."
        ),
        "shipping info": (
            "Free shipping on orders over $50. Delivery in 3-5 business days."
        ),
        "technical support": (
            "Technical support is available 24/7 via chat or phone."
        ),
        "account issues": (
            "For account access problems, visit Settings > Account "
            "or contact support@example.com."
        ),
        "billing": (
            "Billing support is available Mon–Fri 9am–6pm. "
            "View invoices under Settings > Billing."
        ),
    }

    words = query.lower().split()
    results = [
        {"topic": key, "content": article}
        for key, article in kb.items()
        if any(w in key for w in words)
    ]

    if not results:
        return {
            "status": "success",
            "report": f'No articles found matching "{query}"',
            "results": [],
        }

    return {
        "status": "success",
        "report": f'Found {len(results)} article(s) matching "{query}"',
        "results": results,
    }


def create_ticket(
    issue: str,
    tool_context: ToolContext,
    priority: str = "medium",
    customer_email: str = "user@example.com",
) -> Dict[str, Any]:
    """Create a customer support ticket.

    Args:
        issue:          Description of the issue.
        priority:       Priority level — low, medium, or high.
        customer_email: Customer email address.
    """
    if priority not in ("low", "medium", "high"):
        return {
            "status": "error",
            "error": f"Invalid priority: {priority}. Must be low, medium, or high.",
        }

    # Deterministic ticket ID based on issue content
    digest = hashlib.md5(issue.encode()).hexdigest()
    ticket_id = f"TICK-{int(digest, 16) % 10000:04d}"

    estimated_response = "24 hours" if priority == "high" else "48 hours"

    ticket_data = {
        "ticket_id": ticket_id,
        "issue": issue,
        "priority": priority,
        "customer_email": customer_email,
        "status": "open",
        "estimated_response": estimated_response,
    }

    # Persist so check_ticket_status can find it later (same session)
    store = _get_ticket_store(tool_context)
    store[ticket_id] = ticket_data

    return {
        "status": "success",
        "report": f"Ticket {ticket_id} created successfully with {priority} priority",
        "ticket": ticket_data,
    }


def check_ticket_status(ticket_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Check the status of an existing support ticket.

    Args:
        ticket_id: Ticket ID (e.g. TICK-1234).
    """
    # Pre-seeded demo tickets
    static_tickets: Dict[str, Any] = {
        "TICK-1234": {
            "ticket_id": "TICK-1234",
            "status": "open",
            "priority": "high",
            "assigned_to": "Agent Smith",
        },
        "TICK-5678": {
            "ticket_id": "TICK-5678",
            "status": "resolved",
            "priority": "low",
            "resolved_at": "2024-01-15",
        },
    }

    # Merge with dynamically created tickets from this session
    dynamic_tickets = _get_ticket_store(tool_context)
    all_tickets = {**static_tickets, **dynamic_tickets}

    if ticket_id not in all_tickets:
        return {
            "status": "error",
            "error": f"Ticket {ticket_id} not found",
        }

    ticket = all_tickets[ticket_id]
    return {
        "status": "success",
        "report": f"Ticket {ticket_id} status: {ticket['status']}",
        "ticket": ticket,
    }


# ── Agent Definition ─────────────────────────────────────────────────────────

root_agent = Agent(
    name="support_agent",
    model=MODEL,
    description=(
        "Customer support agent that can search the knowledge base, "
        "create tickets, and check ticket status. Designed for systematic testing."
    ),
    instruction="""
    You are a helpful customer support agent.

    CAPABILITIES:
    - Search the knowledge base for answers to common questions
    - Create support tickets for issues
    - Check status of existing tickets

    WORKFLOW:
    1. For questions, search the knowledge base FIRST
    2. If the KB has the answer, provide it directly
    3. If the KB has no answer, or the issue needs follow-up, create a ticket
    4. For ticket status inquiries, use check_ticket_status

    RESPONSE FORMAT:
    - Be concise and professional
    - Always confirm actions (e.g. "I've created ticket TICK-1234")
    - Provide clear next steps

    IMPORTANT:
    - Call search_knowledge_base before creating tickets
    - Use correct priority levels: low, medium, high
    - Always include customer email when creating tickets
    """,
    tools=[search_knowledge_base, create_ticket, check_ticket_status],
    output_key="support_response",
)
