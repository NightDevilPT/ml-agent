"""
Human-in-the-Loop Prompt Utility
=================================

Provides a single professional function to pause automated workflows
and request human input with styled terminal prompts.

Usage:
    from utils.hitl import ask_human

    result = ask_human(
        options={"approve": "Approve changes", "reject": "Reject changes"},
        title="Code Review",
        description="Please review the generated test file.",
        style="warning"
    )
"""

from typing import Optional, Literal, Dict
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def ask_human(
    options: Dict[str, str],
    title: str = "Human Input Required",
    description: Optional[str] = None,
    default: Optional[str] = None,
    style: Literal["info", "warning", "error", "success"] = "warning"
) -> str:
    """
    Pause workflow and ask human to select from multiple options.

    Parameters
    ----------
    options : Dict[str, str]
        Map of option keys to descriptions.
        Example: {"yes": "Approve and continue", "no": "Reject and stop"}

    title : str, optional
        Heading in the prompt panel. Default "Human Input Required".

    description : Optional[str], optional
        Extra context below title. Supports multi-line. Default None.

    default : Optional[str], optional
        Pre-selected key highlighted with arrow (→). Default None (first option).

    style : Literal["info", "warning", "error", "success"], optional
        Visual theme: info(blue), warning(yellow), error(red), success(green).
        Default "warning".

    Returns
    -------
    str
        Selected option key. Example: "approve"

    Raises
    ------
    ValueError
        If options empty or default key not in options.

    Examples
    --------
    >>> result = ask_human(
    ...     options={"approve": "Approve", "reject": "Reject"},
    ...     title="Review Plan",
    ...     style="info"
    ... )
    >>> print(result)
    "approve"
    """
    
    valid_keys = list(options.keys())
    
    if not valid_keys:
        raise ValueError("Options dictionary cannot be empty")
    
    if default is None:
        default = valid_keys[0]
    elif default not in valid_keys:
        raise ValueError(
            f"Default key '{default}' not found. "
            f"Valid keys: {', '.join(valid_keys)}"
        )
    
    style_colors = {
        "info": "blue",
        "warning": "yellow",
        "success": "green",
        "error": "red"
    }
    color = style_colors.get(style, "yellow")
    
    console.print()
    console.print(Panel.fit(
        f"[bold {color}]||  {title}[/bold {color}]",
        border_style=color
    ))
    
    if description:
        console.print(f"[dim]{description}[/dim]")
        console.print()
    
    for key, desc in options.items():
        prefix = "->" if key == default else "  "
        console.print(
            f"  {prefix} [bold green]{key}[/bold green] - [white]{desc}[/white]"
        )
    
    console.print()
    
    choice = Prompt.ask(
        "[bold]Your choice[/bold]",
        choices=valid_keys,
        default=default,
        show_default=True,
        show_choices=False
    )
    
    console.print(f"[dim]Selected: {options[choice]}[/dim]")
    
    return choice