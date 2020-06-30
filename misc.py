"""
"""
from typing import Union


__all__ = [
    "dict_to_str",
]


def dict_to_str(d:Union[dict, list, tuple], current_depth:int=1, indent_spaces:int=4) -> str:
    """
    """
    assert isinstance(d, (dict, list, tuple))
    if len(d) == 0:
        s = f"{{}}" if isinstance(d, dict) else f"[]"
        return s
    s = "\n"
    unit_indent = " "*indent_spaces
    prefix = unit_indent*current_depth
    if isinstance(d, (list, tuple)):
        for v in d:
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{dict_to_str(v, current_depth+1)}\n"
            else:
                s += f"{prefix}{v}\n"
    elif isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{k}: {dict_to_str(v, current_depth+1)}\n"
            else:
                s += f"{prefix}{k}: {v}\n"
    s += unit_indent*(current_depth-1)
    s = f"{{{s}}}" if isinstance(d, dict) else f"[{s}]"
    return s
