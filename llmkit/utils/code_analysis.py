import re

def get_start_line(full_text, snippet):
    """
    Get the starting line number of a snippet within the full text.
    """
    return full_text.count('\n', 0, full_text.index(snippet)) + 1

def get_end_line(full_text, snippet):
    """
    Get the ending line number of a snippet within the full text.
    """
    return get_start_line(full_text, snippet) + snippet.count('\n')

def extract_function_name(code_snippet):
    """
    Extract the function name from a code snippet.
    """
    match = re.search(r'def\s+(\w+)\s*\(', code_snippet)
    return match.group(1) if match else None

def extract_class_name(code_snippet):
    """
    Extract the class name from a code snippet.
    """
    match = re.search(r'class\s+(\w+)\s*:', code_snippet)
    return match.group(1) if match else None