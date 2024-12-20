import math
import re
from typing import Any

import numexpr
from humanlayer import HumanLayer
from langchain_core.tools import BaseTool, tool


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


def list_linear_teams_func() -> Any:
    """
    List all teams in linear.
    """
    return [
        {"id": "team-sales", "name": "Sales"},
        {"id": "team-operations", "name": "Operations"},
    ]
    return {}


def list_linear_people_func(team_id: str) -> Any:
    """
    List all people in linear.
    """
    if team_id == "team-sales":
        return [
            {"id": "person-sales-1", "name": "John Doe"},
            {"id": "person-sales-2", "name": "Jane Doe"},
        ]
    elif team_id == "team-operations":
        return [
            {"id": "person-operations-1", "name": "Austin Alter"},
            {"id": "person-operations-2", "name": "Dexter Horthy"},
        ]
    return {}


def list_linear_projects_func(team_id: str) -> Any:
    """
    List all projects in linear.
    """
    if team_id == "team-sales":
        return [
            {"id": "project-sales-1", "name": "Deals"},
            {"id": "project-sales-2", "name": "Customer Success"},
        ]
    elif team_id == "team-operations":
        return [
            {"id": "project-operations-1", "name": "Taxes"},
            {"id": "project-operations-2", "name": "Upkeep"},
        ]
    return []


hl = HumanLayer()


def add_to_linear_func(
    title: str,
    description: str,
    team_id: str,
    owner_id: str | None,
    project_id: str | None,
) -> str:
    """
    Add a task to linear.
    """
    if team_id not in ["team-sales", "team-operations"]:
        return "unknown team"
    if owner_id not in [
        "person-sales-1",
        "person-sales-2",
        "person-operations-1",
        "person-operations-2",
    ]:
        return "unknown owner"
    if project_id not in [
        "project-sales-1",
        "project-sales-2",
        "project-operations-1",
        "project-operations-2",
    ]:
        return "unknown project"
    return "added to linear"


calculator: BaseTool = tool(calculator_func)
calculator.name = "calculator"

add_to_linear: BaseTool = tool(add_to_linear_func)
add_to_linear.name = "add_to_linear"

list_linear_teams: BaseTool = tool(list_linear_teams_func)
list_linear_teams.name = "list_linear_teams"

list_linear_people: BaseTool = tool(list_linear_people_func)
list_linear_people.name = "list_linear_people"

list_linear_projects: BaseTool = tool(list_linear_projects_func)
list_linear_projects.name = "list_linear_projects"
