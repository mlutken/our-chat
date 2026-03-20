import torch
from pathlib import Path

def parse_between(text, start, end, stop_before_end = False):
    p_start = text.find(start)
    if p_start == -1:
        return -1 , -1

    p_start = p_start + len(start)
    p_end = len(text)
    if end == "":
        return p_start, p_end

    p_end = text.find(end, p_start)
    if p_end == -1:
        return -1 , -1

    if not stop_before_end:
        p_end = p_end + len(end)

    return p_start, p_end


def format_prompt_response(prompt, response, context = ""):
    if context == "":
        return f"<prompt> {prompt} </prompt>\n<response> {response} </response>\n"
    else:
        return f"<context> {context} </context>\n<prompt> {prompt} </prompt>\n<response> {response} </response>\n"


def model_name_from_path(path):
    return Path(path).stem

def save_path_from_model_name(path):
    return Path(path).with_suffix(".pth")
