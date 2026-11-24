"""
PHASE 3: AGENTIC AI ENGINEERING
===============================

BEGINNER GUIDE:

An AI agent is an application that can:
1. Receive a task.
2. Decide which tool to use.
3. Execute that tool.
4. Observe the result.
5. Continue until a stopping condition is reached.

Topics:
- Tool calling
- Agent loop
- ReAct pattern
- Planning
- Reflection
- Short-term and long-term memory
- MCP concepts
- LangGraph concepts
- Multi-agent orchestration
- Human approval
- Checkpointing
- Context engineering
- A2A concepts

Run:
python phase_3_agentic_ai.py
"""

import json
from pathlib import Path


# ------------------------------------------------------------
# 1. TOOL REGISTRY
# ------------------------------------------------------------
# A tool is an ordinary Python function an agent can call.

def calculator(expression):

    # Only allow simple arithmetic characters.
    allowed = set("0123456789+-*/(). ")

    if not expression or any(ch not in allowed for ch in expression):
        return "Invalid mathematical expression."

    try:
        # This calculator intentionally supports only simple arithmetic.
        import ast
        import operator

        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv
        }

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)

            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value

            if isinstance(node, ast.BinOp) and type(node.op) in operations:
                return operations[type(node.op)](
                    evaluate(node.left),
                    evaluate(node.right)
                )

            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -evaluate(node.operand)

            raise ValueError("Unsupported operation.")

        tree = ast.parse(expression, mode="eval")
        return evaluate(tree)

    except Exception as error:
        return f"Calculation error: {error}"


def search_knowledge(query):

    knowledge = {
        "rag": "RAG retrieves external information before generating an answer.",
        "agent": "An AI agent uses tools and decisions to complete tasks.",
        "mcp": "MCP is a protocol for connecting AI applications with tools and context.",
        "langgraph": "LangGraph supports stateful graph-based agent workflows."
    }

    for key, value in knowledge.items():
        if key in query.lower():
            return value

    return "No matching knowledge found."


TOOLS = {
    "calculator": calculator,
    "search_knowledge": search_knowledge
}


# ------------------------------------------------------------
# 2. TOOL CALLING
# ------------------------------------------------------------

def call_tool(tool_name, argument):

    if tool_name not in TOOLS:
        return "Tool is not permitted."

    return TOOLS[tool_name](argument)


# ------------------------------------------------------------
# 3. AGENT LOOP
# ------------------------------------------------------------
# The loop allows an agent to perform several steps.
# A maximum step count prevents endless execution.

def agent_loop(task, max_steps=5):

    print("\nAGENT STARTED")
    print("Task:", task)

    for step in range(1, max_steps + 1):

        print(f"\nStep {step}")

        if "calculate" in task.lower():
            expression = task.lower().replace("calculate", "").strip()
            result = call_tool("calculator", expression)

        else:
            result = call_tool("search_knowledge", task)

        print("Tool result:", result)

        # This educational agent stops after one tool result.
        # A real LLM agent would decide whether another tool is needed.
        print("Agent stopped successfully.")
        return result

    return "Maximum agent steps reached."


# ------------------------------------------------------------
# 4. MEMORY
# ------------------------------------------------------------

class AgentMemory:

    def __init__(self):
        self.messages = []

    def remember(self, message):
        self.messages.append(message)

    def recall(self):
        return self.messages


# ------------------------------------------------------------
# 5. HUMAN APPROVAL
# ------------------------------------------------------------

def approved_action(action, approved=False):

    if not approved:
        return "Action blocked: human approval required."

    return f"Approved action: {action}"


# ------------------------------------------------------------
# 6. CHECKPOINTING
# ------------------------------------------------------------
# A checkpoint saves agent state so work can be resumed.

def save_checkpoint(state, filename="agent_checkpoint.json"):

    path = Path(__file__).resolve().parent / filename
    path.write_text(json.dumps(state, indent=4), encoding="utf-8")

    print("Checkpoint saved:", path)


def load_checkpoint(filename="agent_checkpoint.json"):

    path = Path(__file__).resolve().parent / filename

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------
# 7. PLANNING AND REFLECTION
# ------------------------------------------------------------

def create_plan(task):

    return [
        "Understand the task",
        "Identify the required information",
        "Select an appropriate tool",
        "Check the result",
        "Return the final answer"
    ]


def reflect(result):

    if result and "error" not in str(result).lower():
        return "Result received. Basic validation passed."

    return "Result requires further review."


# ------------------------------------------------------------
# 8. MULTI-AGENT ORCHESTRATION
# ------------------------------------------------------------
# Multiple specialist functions can cooperate.
# This is a simple demonstration, not an autonomous LLM team.

def research_agent(question):
    return search_knowledge(question)


def reviewer_agent(answer):
    return {
        "answer": answer,
        "reviewed": bool(answer)
    }


def orchestrator(question):

    research = research_agent(question)
    review = reviewer_agent(research)

    return review


# ------------------------------------------------------------
# 9. MCP CONCEPT
# ------------------------------------------------------------
# This demonstrates MCP-style tool discovery in Python.
# It is NOT a real networked MCP server.
# For actual MCP, implement the official SDK and transport.

def mcp_tool_discovery():

    server_description = {
        "server": "Learning MCP Server",
        "tools": [
            {
                "name": "search_knowledge",
                "description": "Search educational knowledge."
            },
            {
                "name": "calculator",
                "description": "Perform simple arithmetic."
            }
        ]
    }

    return server_description


# ------------------------------------------------------------
# 10. LANGGRAPH CONCEPT
# ------------------------------------------------------------
# LangGraph models workflows as nodes and edges.
# This simple example demonstrates state transitions.

def simple_graph(question):

    state = {
        "question": question,
        "result": None,
        "status": "START"
    }

    state["result"] = search_knowledge(question)
    state["status"] = "COMPLETED"

    return state


# ------------------------------------------------------------
# 11. MAIN
# ------------------------------------------------------------

def main():

    print("PHASE 3: AGENTIC AI ENGINEERING")

    print("\n1. TOOL CALLING")
    print(call_tool("calculator", "10 + 20 * 2"))

    print("\n2. AGENT LOOP")
    agent_loop("Explain RAG")

    print("\n3. MEMORY")

    memory = AgentMemory()
    memory.remember("The user is learning Agentic AI.")
    memory.remember("The user has completed RAG fundamentals.")

    print(memory.recall())

    print("\n4. HUMAN APPROVAL")
    print(approved_action("Send an email", approved=False))

    print("\n5. PLANNING")
    print(create_plan("Build an AI research assistant"))

    print("\n6. MULTI-AGENT ORCHESTRATION")
    print(orchestrator("Explain LangGraph"))

    print("\n7. MCP TOOL DISCOVERY")
    print(mcp_tool_discovery())

    print("\n8. GRAPH WORKFLOW")
    print(simple_graph("What is RAG?"))

    print("\n9. CHECKPOINT")
    save_checkpoint({
        "task": "Learn Agentic AI",
        "status": "completed"
    })

    print("Loaded checkpoint:", load_checkpoint())

    print("\nPHASE 3 DEMONSTRATION COMPLETED")


if __name__ == "__main__":
    main()


# INTERVIEW QUESTIONS
#
# 1. What is an AI agent?
# 2. What is tool calling?
# 3. Explain the ReAct pattern.
# 4. What is the agent loop?
# 5. Why is memory useful?
# 6. What is LangGraph?
# 7. What is MCP?
# 8. Explain human-in-the-loop.
# 9. What is multi-agent orchestration?
# 10. What are agent checkpoints?