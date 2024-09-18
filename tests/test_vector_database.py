import pytest
import pytest_asyncio
from llmkit.vector_databases.vector_database_base import VectorDatabase

class MockVectorDatabase(VectorDatabase):
    async def add_vector(self, vector, metadata):
        pass

    async def search_vectors(self, query_vector, top_k=5):
        return []

@pytest.mark.asyncio
async def test_vector_database_interface():
    db = MockVectorDatabase()
    await db.add_vector([0.1, 0.2, 0.3], {"content": "Test"})
    results = await db.search_vectors([0.1, 0.2, 0.3])
    assert results == []