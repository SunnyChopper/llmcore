import pytest
from llmkit.memory import MemoryManager
from llmkit.core import LLMConfig

@pytest.mark.asyncio
async def test_memory_manager_add_and_retrieve():
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    memory = {"vector": [0.1, 0.2, 0.3], "content": "Test memory"}
    
    await memory_manager.add_memory(memory)
    assert len(memory_manager.memories) == 1
    assert memory_manager.memories[0] == memory
    
    query_vector = [0.1, 0.2, 0.3]
    relevant_memories = await memory_manager.get_relevant_memories(query_vector)
    assert len(relevant_memories) == 1
    assert relevant_memories[0]["content"] == "Test memory"

@pytest.mark.asyncio
async def test_memory_manager_clear_memories():
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    memory1 = {"vector": [0.1, 0.2, 0.3], "content": "Test memory 1"}
    memory2 = {"vector": [0.4, 0.5, 0.6], "content": "Test memory 2"}
    
    await memory_manager.add_memory(memory1)
    await memory_manager.add_memory(memory2)
    assert len(memory_manager.memories) == 2
    
    memory_manager.clear()
    assert len(memory_manager.memories) == 0

@pytest.mark.asyncio
async def test_memory_manager_get_relevant_memories_with_threshold():
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    memory1 = {"vector": [0.1, 0.2, 0.3], "content": "Test memory 1"}
    memory2 = {"vector": [0.4, 0.5, 0.6], "content": "Test memory 2"}
    memory3 = {"vector": [0.7, 0.8, 0.9], "content": "Test memory 3"}
    
    await memory_manager.add_memory(memory1)
    await memory_manager.add_memory(memory2)
    await memory_manager.add_memory(memory3)
    
    query_vector = [0.11, 0.19, 0.31]  # Near match with memory1
    relevant_memories = await memory_manager.get_relevant_memories(query_vector, threshold=0.99)
    assert len(relevant_memories) == 1
    assert relevant_memories[0]["content"] == "Test memory 1"
    
    # Test with a lower threshold to get multiple results
    relevant_memories = await memory_manager.get_relevant_memories(query_vector, threshold=0.9)
    assert len(relevant_memories) > 1
    assert relevant_memories[0]["content"] == "Test memory 1"
    
    # Test with a very high threshold to get no results
    relevant_memories = await memory_manager.get_relevant_memories(query_vector, threshold=1.1)
    assert len(relevant_memories) == 0