import pytest
import os
import aiohttp
from unittest.mock import AsyncMock, patch
from llmcore.core import LLM, LLMAPIError, LLMJSONParseError, LLMNetworkError
from llmcore.config import LLMConfig
from llmcore.prompt import Prompt, PromptTemplate
from llmcore.contracts import ConversationTurn

@pytest.fixture
def mock_env_vars():
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "mock_openai_key",
        "ANTHROPIC_API_KEY": "mock_anthropic_key",
        "GEMINI_API_KEY": "mock_google_key"
    }):
        yield

@pytest.fixture
def mock_openai_client():
    with patch('llmcore.core.OpenAIClientAdapter') as mock:
        yield mock

@pytest.fixture
def mock_anthropic_client():
    with patch('llmcore.core.AnthropicClientAdapter') as mock:
        yield mock

@pytest.fixture
def mock_google_client():
    with patch('llmcore.core.GoogleGeminiClientAdapter') as mock:
        yield mock

@pytest.fixture
def llm_openai(mock_env_vars, mock_openai_client):
    llm = LLM('openai', 'gpt-4o-mini', LLMConfig(temperature=0.7, max_tokens=100))
    llm.client = mock_openai_client.return_value
    return llm

@pytest.fixture
def llm_anthropic(mock_env_vars, mock_anthropic_client):
    llm = LLM('anthropic', 'claude-3-5-sonnet-20240620', LLMConfig(temperature=0.5, max_tokens=150))
    llm.client = mock_anthropic_client.return_value
    return llm

@pytest.fixture
def llm_google(mock_env_vars, mock_google_client):
    llm = LLM('google', 'gemini-1.5-pro', LLMConfig(temperature=0.3, max_tokens=200))
    llm.client = mock_google_client.return_value
    return llm

def test_llm_initialization(llm_openai, llm_anthropic, llm_google):
    assert llm_openai.provider == 'openai'
    assert llm_openai.model == 'gpt-4o-mini'
    assert llm_openai.config.temperature == 0.7
    assert llm_openai.config.max_tokens == 100

    assert llm_anthropic.provider == 'anthropic'
    assert llm_anthropic.model == 'claude-3-5-sonnet-20240620'
    assert llm_anthropic.config.temperature == 0.5
    assert llm_anthropic.config.max_tokens == 150

    assert llm_google.provider == 'google'
    assert llm_google.model == 'gemini-1.5-pro'
    assert llm_google.config.temperature == 0.3
    assert llm_google.config.max_tokens == 200

def test_llm_initialization_invalid_provider(mock_env_vars):
    import re
    with pytest.raises(ValueError, match=re.escape("API key (INVALID_PROVIDER_API_KEY) for invalid_provider not found in system or user environment variables")):
        LLM('invalid_provider', 'model')

@pytest.mark.asyncio
async def test_send_input_async(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}))
    llm_openai.client.send_prompt = AsyncMock(return_value="Test response")
    
    response = await llm_openai.send_input_async(prompt)
    assert isinstance(response, str)
    assert response == "Test response"

    llm_openai.client.send_prompt.assert_called_once_with("Test prompt", llm_openai.config)

@pytest.mark.asyncio
async def test_send_input_async_with_json_parsing(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}, output_json_structure={"key": str}))
    llm_openai.client.send_prompt = AsyncMock(return_value='{"key": "value"}')
    
    response = await llm_openai.send_input_async(prompt, parse_json=True)
    assert isinstance(response, dict)
    assert response == {"key": "value"}

@pytest.mark.asyncio
async def test_stream_input_async(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}))

    async def mock_stream():
        yield "Test "
        yield "response"

    # Create a proper async generator
    async def mock_stream_prompt(*args, **kwargs):
        async for chunk in mock_stream():
            yield chunk

    # Mock the stream_prompt method with our async generator
    llm_openai.client.stream_prompt = mock_stream_prompt

    chunks = []
    async for chunk in llm_openai.stream_input_async(prompt):
        chunks.append(chunk)

    assert chunks == ["Test ", "response"]

@pytest.mark.asyncio
async def test_send_input_with_memory(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}))
    llm_openai.memory_manager.get_relevant_memories = AsyncMock(return_value=[{"content": "Memory 1"}, {"content": "Memory 2"}])
    llm_openai.client.send_prompt = AsyncMock(return_value="Test response")
    
    response = await llm_openai.send_input_with_memory(prompt)
    assert response == "Test response"
    llm_openai.client.send_prompt.assert_called_once()

@pytest.mark.asyncio
async def test_send_input_with_history(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}))
    history = [
        ConversationTurn("Human", "Hello"),
        ConversationTurn("AI", "Hi there!")
    ]
    
    llm_openai.client.send_prompt = AsyncMock(return_value="Test response with history")
    
    response = await llm_openai.send_input_with_history(prompt, history)
    assert response == "Test response with history"
    
    expected_prompt = "Human: Hello\nAI: Hi there!\n\nHuman: Test prompt\nAI:"
    llm_openai.client.send_prompt.assert_called_once_with(expected_prompt, llm_openai.config)

def test_update_config(llm_openai):
    llm_openai.update_config(temperature=0.8, max_tokens=200)
    assert llm_openai.config.temperature == 0.8
    assert llm_openai.config.max_tokens == 200

    with pytest.raises(ValueError, match="Invalid configuration option: invalid_option"):
        llm_openai.update_config(invalid_option=1)

@pytest.mark.asyncio
async def test_llm_api_error(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}))
    llm_openai.client.send_prompt = AsyncMock(side_effect=Exception("API Error"))
    
    with pytest.raises(LLMAPIError, match="Unexpected error occurred while sending prompt to LLM: API Error"):
        await llm_openai.send_input_async(prompt)

@pytest.mark.asyncio
async def test_parse_json_response_valid(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}, output_json_structure={"key": str, "value": int}))
    llm_openai.client.send_prompt = AsyncMock(return_value='{"key": "test", "value": 42}')
    
    response = await llm_openai.send_input_async(prompt, parse_json=True)
    assert response == {"key": "test", "value": 42}

@pytest.mark.asyncio
async def test_parse_json_response_invalid(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}, output_json_structure={"key": str}))
    llm_openai.client.send_prompt = AsyncMock(return_value='Invalid JSON')
    
    with pytest.raises(LLMJSONParseError, match="LLM did not return a valid JSON object"):
        await llm_openai.send_input_async(prompt, parse_json=True)

@pytest.mark.asyncio
async def test_llm_network_error(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}))
    llm_openai.client.send_prompt = AsyncMock(side_effect=aiohttp.ClientError("Network Error"))
    
    with pytest.raises(LLMNetworkError, match="Network error occurred while sending prompt to LLM"):
        await llm_openai.send_input_async(prompt)

@pytest.mark.asyncio
async def test_configure_json_mode(llm_openai):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}, output_json_structure={"key": str}))
    llm_openai.client.send_prompt = AsyncMock(return_value='{"key": "value"}')
    
    await llm_openai.send_input_async(prompt, parse_json=True)
    
    assert llm_openai.config.response_format == {"type": "json_object"}

@pytest.mark.asyncio
async def test_configure_json_mode_anthropic(llm_anthropic):
    prompt = Prompt(PromptTemplate("Test prompt", required_params={}, output_json_structure={"key": str}))
    llm_anthropic.client.send_prompt = AsyncMock(return_value='{"key": "value"}')
    
    await llm_anthropic.send_input_async(prompt, parse_json=True)
    
    assert llm_anthropic.config.json_response == True
    assert hasattr(llm_anthropic.config, 'json_instruction')
    if "claude-3" in llm_anthropic.model:
        assert llm_anthropic.config.json_instruction == LLM.JSON_ENSURE_RESPONSE