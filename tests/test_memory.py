import pytest
from unittest.mock import AsyncMock, patch
from llmcore.memory import MemoryManager
from llmcore.core import LLMConfig, RelevantMemory

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
    assert relevant_memories[0].content == "Test memory"

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
    
    query_vector = [0.12, 0.18, 0.32]  # Near match with memory1
    relevant_memories = await memory_manager.get_relevant_memories(query_vector, threshold=0.99)
    assert len(relevant_memories) == 1
    assert relevant_memories[0].content == "Test memory 1"
    
    # Test with a lower threshold to get multiple results
    relevant_memories = await memory_manager.get_relevant_memories(query_vector, threshold=0.9)
    assert len(relevant_memories) > 1
    assert relevant_memories[0].content == "Test memory 1"
    
    # Test with a very high threshold to get no results
    relevant_memories = await memory_manager.get_relevant_memories(query_vector, threshold=0.999)
    assert len(relevant_memories) == 0

@pytest.mark.asyncio
async def test_add_memory_with_invalid_vector():
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    invalid_memories = [
        {"vector": "not_a_vector", "content": "Invalid memory"},  # Vector is a string
        {"content": "Missing vector key"},                       # Missing 'vector' key
        {"vector": None, "content": "None vector"},             # Vector is None
        {"vector": [0.1, "invalid", 0.3], "content": "Mixed types"}  # Mixed types in vector
    ]
    
    for mem in invalid_memories:
        with pytest.raises((ValueError, TypeError)):  # Accept both exception types
            await memory_manager.add_memory(mem)

@pytest.mark.asyncio
async def test_add_memory_with_vector_db_success():
    from llmcore.memory import Vector
    config = LLMConfig(vector_db_provider="pinecone", vector_db_endpoint="http://localhost", vector_db_api_key="fake_key")
    with patch('llmcore.memory.PineconeDatabase.add_vector', new_callable=AsyncMock) as mock_add_vector:
        memory_manager = MemoryManager(config)
        memory = {"vector": [0.1, 0.2, 0.3], "content": "Test memory with DB"}
        
        await memory_manager.add_memory(memory)
        mock_add_vector.assert_awaited_once_with([0.1, 0.2, 0.3], {"content": "Test memory with DB"})
        assert len(memory_manager.memories) == 1
        # Separate assertions for 'content' and 'vector'
        assert memory_manager.memories[0]["content"] == "Test memory with DB"
        # Compare vector values directly
        stored_vector = memory_manager.memories[0]["vector"]
        assert isinstance(stored_vector, Vector)
        assert stored_vector.tolist() == [0.1, 0.2, 0.3]

@pytest.mark.asyncio
async def test_add_memory_with_vector_db_key_error():
    config = LLMConfig(vector_db_provider="pinecone", vector_db_endpoint="http://localhost", vector_db_api_key="fake_key")
    with patch('llmcore.memory.PineconeDatabase.add_vector', new_callable=AsyncMock) as mock_add_vector:
        mock_add_vector.side_effect = KeyError("missing_content")
        memory_manager = MemoryManager(config)
        memory = {"vector": [0.1, 0.2, 0.3]}  # Missing 'content' key
        
        with pytest.raises(ValueError, match="Memory dict is missing required key: 'content'"):
            await memory_manager.add_memory(memory)

@pytest.mark.asyncio
async def test_add_memory_with_vector_db_exception():
    config = LLMConfig(vector_db_provider="pinecone", vector_db_endpoint="http://localhost", vector_db_api_key="fake_key")
    with patch('llmcore.memory.PineconeDatabase.add_vector', new_callable=AsyncMock) as mock_add_vector:
        mock_add_vector.side_effect = Exception("DB connection error")
        memory_manager = MemoryManager(config)
        memory = {"vector": [0.1, 0.2, 0.3], "content": "Test memory with DB exception"}
        
        with pytest.raises(RuntimeError, match="Failed to add vector to database: DB connection error"):
            await memory_manager.add_memory(memory)

@pytest.mark.asyncio
async def test_get_relevant_memories_without_vector_db():
    config = LLMConfig(vector_db_provider=None)
    memory_manager = MemoryManager(config)
    memory1 = {"vector": [0.1, 0.2, 0.3], "content": "Memory 1"}
    memory2 = {"vector": [0.2, 0.3, 0.4], "content": "Memory 2"}
    memory3 = {"vector": "invalid_vector", "content": "Memory 3"}  # Should be skipped
    
    await memory_manager.add_memory(memory1)
    await memory_manager.add_memory(memory2)
    
    # Attempt to add memory3 and expect a ValueError
    with pytest.raises(ValueError, match="Memory 'vector' must be a list of floats or a Vector."):
        await memory_manager.add_memory(memory3)
    
    query_vector = [0.15, 0.25, 0.35]
    relevant_memories = await memory_manager.get_relevant_memories(query_vector)
    assert len(relevant_memories) == 2
    assert set(memory.content for memory in relevant_memories) == {"Memory 1", "Memory 2"}

@pytest.mark.asyncio
async def test_capacity_enforcement():
    config = LLMConfig()
    memory_manager = MemoryManager(config, capacity=3)
    memories = [
        {"vector": [0.1, 0.2, 0.3], "content": "Memory 1"},
        {"vector": [0.2, 0.3, 0.4], "content": "Memory 2"},
        {"vector": [0.3, 0.4, 0.5], "content": "Memory 3"},
        {"vector": [0.4, 0.5, 0.6], "content": "Memory 4"}
    ]
    
    for mem in memories:
        await memory_manager.add_memory(mem)
    
    assert len(memory_manager.memories) == 3
    assert memory_manager.memories[0]["content"] == "Memory 2"
    assert memory_manager.memories[1]["content"] == "Memory 3"
    assert memory_manager.memories[2]["content"] == "Memory 4"

@pytest.mark.asyncio
async def test_calculate_similarity():
    from llmcore.memory import Vector
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    
    vector1 = Vector([1, 0, 0])
    vector2 = Vector([0, 1, 0])
    vector3 = Vector([1, 1, 0])
    
    sim1 = memory_manager._calculate_similarity(vector1, vector2)
    assert sim1 == 0.0  # Orthogonal vectors
    
    sim2 = memory_manager._calculate_similarity(vector1, vector3)
    assert abs(sim2 - 0.7071) < 0.0001  # 45 degrees

@pytest.mark.asyncio
async def test_clear_method_with_vector_db():
    config = LLMConfig(vector_db_provider="pinecone", vector_db_endpoint="http://localhost", vector_db_api_key="fake_key")
    with patch('llmcore.memory.PineconeDatabase') as mock_pinecone_db:
        mock_db_instance = mock_pinecone_db.return_value
        # Ensure that add_vector is an AsyncMock to make it awaitable
        mock_db_instance.add_vector = AsyncMock()
        memory_manager = MemoryManager(config)
        memory1 = {"vector": [0.1, 0.2, 0.3], "content": "Memory 1"}
        memory2 = {"vector": [0.4, 0.5, 0.6], "content": "Memory 2"}
        
        await memory_manager.add_memory(memory1)
        await memory_manager.add_memory(memory2)
        assert len(memory_manager.memories) == 2
        
        memory_manager.clear()
        assert len(memory_manager.memories) == 0
        # Assuming there's a method to clear the vector DB if supported
        mock_db_instance.clear.assert_not_called()  # Since it's not implemented

@pytest.mark.asyncio
async def test_get_relevant_memories_empty():
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    query_vector = [0.1, 0.2, 0.3]
    relevant_memories = await memory_manager.get_relevant_memories(query_vector)
    assert relevant_memories == []

@pytest.mark.asyncio
async def test_add_memory_with_large_vector():
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    large_vector = list(range(1000))  # Large vector
    memory = {"vector": large_vector, "content": "Large vector memory"}
    
    await memory_manager.add_memory(memory)
    assert len(memory_manager.memories) == 1
    assert memory_manager.memories[0]["vector"].tolist() == large_vector
    assert memory_manager.memories[0]["content"] == "Large vector memory"

@pytest.mark.asyncio
async def test_get_relevant_memories_with_no_matching_threshold():
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    memory = {"vector": [0.9, 0.9, 0.9], "content": "Distant memory"}
    
    await memory_manager.add_memory(memory)
    query_vector = [0.06, 0.15, 0.09]
    
    relevant_memories = await memory_manager.get_relevant_memories(query_vector, threshold=0.99)
    assert len(relevant_memories) == 0

@pytest.mark.asyncio
async def test_get_relevant_memories_invalid_query_vector():
    config = LLMConfig()
    memory_manager = MemoryManager(config)
    memory = {"vector": [0.1, 0.2, 0.3], "content": "Valid memory"}
    
    await memory_manager.add_memory(memory)
    
    invalid_query_vectors = [
        ("not_a_vector", TypeError, "Vector data must be a list of floats or a Vector."),
        ([0.1, 0.2], ValueError, "Query vector must have dimension 3."),
        (None, TypeError, "Vector data must be a list of floats or a Vector."),
        ({"x": 1, "y": 2}, TypeError, "Vector data must be a list of floats or a Vector."),
    ]
    
    for query, expected_exception, expected_message in invalid_query_vectors:
        with pytest.raises(expected_exception, match=expected_message):
            await memory_manager.get_relevant_memories(query)
