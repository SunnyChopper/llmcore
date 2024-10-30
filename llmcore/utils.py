from typing import List
from math import sqrt

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    # Compute dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Compute magnitudes
    norm1 = sqrt(sum(x * x for x in v1))
    norm2 = sqrt(sum(x * x for x in v2))
    
    # Check for zero magnitude
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("One or both vectors have zero magnitude")
        
    # Calculate similarity and clamp to [-1.0, 1.0] to handle floating point errors
    similarity = dot_product / (norm1 * norm2)
    return max(min(similarity, 1.0), -1.0)