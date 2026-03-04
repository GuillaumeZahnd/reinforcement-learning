from minigrid.core.constants import IDX_TO_OBJECT, OBJECT_TO_IDX, STATE_TO_IDX, COLOR_TO_IDX
from minigrid.core.actions import Actions


IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}


def action_idx2name(idx: int) -> str:
    """
    Actions:
    0: left
    1: right
    2: forward
    3: pickup
    4: drop
    5: toggle
    6: done
    """
    try:
        return Actions(idx).name
    except ValueError:
        return f"Unknown action idx: {idx}."


def action_name2idx(name: str) -> int:
    """
    Actions:
    0: left
    1: right
    2: forward
    3: pickup
    4: drop
    5: toggle
    6: done
    """
    try:
        return Actions[name.lower()].value
    except ValueError:
        return f"Unknown action name: {name}."


def object_idx2name(idx: int) -> str:
    """
    Objects:
    0: unseen
    1: empty
    2: wall
    4: door
    5: key
    8: goal
    """
    try:
        return IDX_TO_OBJECT[idx]
    except KeyError:
        return f"Unknown object idx: {idx}"


def object_name2idx(name: str) -> int:
    """
    Objects:
    0: unseen
    1: empty
    2: wall
    4: door
    5: key
    8: goal
    """
    try:
        return OBJECT_TO_IDX[name.lower()]
    except KeyError:
        return f"Unknown object name: {idx}"

def state_idx2name(idx: int) -> str:
    """
    States:
    0: open
    1: closed
    2: locked
    """
    try:
        return IDX_TO_STATE[idx]
    except KeyError:
        return f"Unknown state idx: {idx}"


def state_name2idx(name: str) -> int:
    """
    States:
    0: open
    1: closed
    2: locked
    """
    try:
        return STATE_TO_IDX[name.lower()]
    except KeyError:
        return f"Unknown state name: {idx}"


def color_idx2name(idx: int) -> str:
    """
    Colors:
    0: red
    1: green
    2: blue
    3: purple
    4: yellow
    5: grey
    """
    try:
        return IDX_TO_STATE[idx]
    except KeyError:
        return f"Unknown color idx: {idx}"


def color_name2idx(name: str) -> int:
    """
    Colors:
    0: red
    1: green
    2: blue
    3: purple
    4: yellow
    5: grey
    """
    try:
        return STATE_TO_IDX[name.lower()]
    except KeyError:
        return f"Unknown color name: {idx}"


def print_green(text: str):
    print("\033[32m" + text + "\033[0m")


def print_red(text: str):
    print("\033[31m" + text + "\033[0m")
