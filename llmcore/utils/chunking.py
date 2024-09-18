def sliding_window_chunk(text, window_size=10, stride=5):
    """
    Split text into chunks using a sliding window approach.
    """
    lines = text.split('\n')
    return ['\n'.join(lines[i:i + window_size]) for i in range(0, len(lines) - window_size + 1, stride)]