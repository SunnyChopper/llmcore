import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from llmcore.chain import LLMChain, LLMChainBuilder, LLMChainError, LLMChainStep
from llmcore.prompt import PromptTemplate

class AsyncIteratorMock:
    def __init__(self, *values):
        self.values = values
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.values):
            value = self.values[self.index]
            self.index += 1
            return value
        else:
            raise StopAsyncIteration

@pytest.fixture
def mock_llm():
    with patch('llmcore.chain.LLM') as MockLLM:
        mock_instance = MockLLM.return_value
        mock_instance.send_input_async = AsyncMock()
        mock_instance.send_input_with_memory = AsyncMock()
        mock_instance.stream_input_async = AsyncMock()
        yield mock_instance

@pytest.fixture
def sample_prompt_template():
    return PromptTemplate(
        template="Hello, {{name}}!",
        required_params={"name": str},
        output_json_structure={"greeting": str}
    )

@pytest.fixture
def sample_step(sample_prompt_template):
    return LLMChainStep(
        prompt_template=sample_prompt_template,
        output_key="greeting"
    )

@pytest.fixture
def llm_chain(mock_llm, sample_step):
    return LLMChain(
        default_llm=mock_llm,
        steps=[sample_step],
        use_memory=False
    )

@pytest.mark.asyncio
async def test_llm_chain_execute_async_success(llm_chain, mock_llm):
    # Arrange
    initial_input = {"name": "Alice"}
    mock_llm.send_input_async.return_value = {"greeting": "Hello, Alice!"}

    # Act
    result = await llm_chain.execute_async(initial_input)

    # Assert
    mock_llm.send_input_async.assert_awaited_once()
    assert result == {"name": "Alice", "greeting": {"greeting": "Hello, Alice!"}}

def test_llm_chain_execute_sync_success(llm_chain, mock_llm):
    # Arrange
    initial_input = {"name": "Bob"}
    mock_llm.send_input_async.return_value = {"greeting": "Hello, Bob!"}

    # Act
    result = llm_chain.execute(initial_input)

    # Assert
    mock_llm.send_input_async.assert_awaited_once()
    assert result == {"name": "Bob", "greeting": {"greeting": "Hello, Bob!"}}

@pytest.mark.asyncio
async def test_llm_chain_execute_async_missing_param(llm_chain):
    # Arrange
    initial_input = {}  # Missing 'name'

    # Act & Assert
    with pytest.raises(LLMChainError) as exc_info:
        await llm_chain.execute_async(initial_input)
    
    assert "Input validation failed for step 1: Missing required parameter: name" in str(exc_info.value)

@pytest.mark.asyncio
async def test_llm_chain_execute_async_invalid_param_type(llm_chain):
    # Arrange
    initial_input = {"name": 123}  # 'name' should be str

    # Act & Assert
    with pytest.raises(LLMChainError) as exc_info:
        await llm_chain.execute_async(initial_input)
    
    assert "Input validation failed for step 1: Parameter name must be of type <class 'str'>" in str(exc_info.value)

@pytest.mark.asyncio
async def test_llm_chain_execute_async_llm_failure(llm_chain, mock_llm):
    # Arrange
    initial_input = {"name": "Charlie"}
    mock_llm.send_input_async.side_effect = Exception("LLM service error")

    # Act & Assert
    with pytest.raises(LLMChainError) as exc_info:
        await llm_chain.execute_async(initial_input)
    
    assert "LLM request failed for step 1: LLM service error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_llm_chain_stream_async_success(llm_chain, mock_llm):
    # Arrange
    initial_input = {"name": "Diana"}

    # Create an instance of AsyncIteratorMock with the desired outputs
    stream_mock = AsyncIteratorMock("Part 1 ", {"greeting": "Hello, Diana!"})

    # Configure the mock_llm's stream_input_async to return the AsyncIteratorMock instance
    mock_llm.stream_input_async = MagicMock(return_value=stream_mock)

    # Act
    output = await llm_chain.stream_async(initial_input).__anext__()
    # Assert each yield
    if isinstance(output, dict):
        assert output == {"greeting": "Hello, Diana!"}

@pytest.mark.asyncio
async def test_llm_chain_extract_json_valid(llm_chain):
    # Arrange
    text = '{"key": "value"}'

    # Act
    result = await llm_chain._extract_json(text)

    # Assert
    assert result == {"key": "value"}

@pytest.mark.asyncio
async def test_llm_chain_extract_json_invalid_then_llm_fallback(llm_chain, mock_llm):
    # Arrange
    text = "Some invalid JSON text"
    mock_llm.send_input_async.return_value = {"extracted_json": {"key": "fallback_value"}}

    # Act
    result = await llm_chain._extract_json(text)

    # Assert
    mock_llm.send_input_async.assert_awaited_once()
    assert result == {"key": "fallback_value"}

@pytest.mark.asyncio
async def test_llm_chain_extract_json_code_block(llm_chain):
    # Arrange
    text = 'Here is some JSON: ```json\n{"key": "value"}\n```'

    # Act
    result = await llm_chain._extract_json(text)

    # Assert
    assert result == {"key": "value"}

@pytest.mark.asyncio
async def test_llm_chain_extract_json_multiple_json(llm_chain):
    # Arrange
    text = 'First JSON: {"key1": "value1"} and second JSON: {"key2": "value2", "key3": "value3"}'

    # Act
    result = await llm_chain._extract_json(text)

    # Assert
    assert result == {"key2": "value2", "key3": "value3"}

@pytest.mark.asyncio
async def test_llm_chain_extract_json_no_json(llm_chain, mock_llm):
    # Arrange
    text = "No JSON here."
    mock_llm.send_input_async.return_value = {"extracted_json": {}}

    # Act
    result = await llm_chain._extract_json(text)

    # Assert
    mock_llm.send_input_async.assert_awaited_once()
    assert result == {}

def test_llm_chain_builder_add_step_and_build(mock_llm):
    # Arrange
    builder = LLMChainBuilder(default_llm=mock_llm)
    prompt_template = PromptTemplate(
        template="Say hi to {{name}}",
        required_params={"name": str},
        output_json_structure={"response": str}
    )
    step = LLMChainStep(prompt_template, "response")

    # Act
    builder.add_step(
        template=prompt_template.template,
        output_key="response",
        required_params=prompt_template.required_params,
        output_json_structure=prompt_template.output_json_structure
    )
    chain = builder.build()

    # Assert
    assert isinstance(chain, LLMChain)
    assert len(chain.steps) == 1
    assert chain.steps[0].prompt_template.template == "Say hi to {{name}}"
    assert chain.steps[0].output_key == "response"

@pytest.mark.asyncio
async def test_llm_chain_execute_with_memory(llm_chain, mock_llm):
    # Arrange
    llm_chain.use_memory = True
    initial_input = {"name": "Eve"}
    mock_llm.send_input_with_memory.return_value = {"greeting": "Hello, Eve!"}

    # Act
    result = await llm_chain.execute_async(initial_input)

    # Assert
    assert result == {"name": "Eve", "greeting": {"greeting": "Hello, Eve!"}}

@pytest.mark.asyncio
async def test_llm_chain_validate_input_nested_dict():
    # Arrange
    prompt_template = PromptTemplate(
        template="Process {{data}}",
        required_params={"data": Dict[str, Any]},
        output_json_structure={"processed_data": Dict[str, Any]}
    )
    step = LLMChainStep(prompt_template, "processed_data")
    chain = LLMChain(
        default_llm=AsyncMock(),
        steps=[step],
        use_memory=False
    )
    initial_input = {"data": {"key1": "value1", "key2": 2}}

    # Mock the LLM response
    chain.default_llm.send_input_async.return_value = {"processed_data": {"key1": "value1", "key2": 2}}

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    chain.default_llm.send_input_async.assert_awaited_once()
    assert result == {"data": {"key1": "value1", "key2": 2}, "processed_data": {"processed_data": {"key1": "value1", "key2": 2}}}

@patch('llmcore.memory.get_vector_database')
def test_llm_chain_builder_add_step_with_dict_llm(mock_get_vector_db, mock_llm):
    # Arrange
    mock_get_vector_db.return_value = MagicMock()  # Mock the VectorDatabase

    builder = LLMChainBuilder()
    step_llm_config = {
        "provider": "google",
        "model": "gemini-1.5",
        "config": {
            "temperature": 0.5,
            "max_tokens": 1000,
            "vector_db_provider": "mock_provider"  # Added key to prevent AttributeError
        }
    }
    prompt_template = PromptTemplate(
        template="Translate {{text}} to {{language}}",
        required_params={"text": str, "language": str},
        output_json_structure={"translation": str}
    )

    # Prepare the mocked LLM instance to have the necessary attributes
    mock_llm.provider = step_llm_config["provider"]
    mock_llm.model = step_llm_config["model"]
    mock_llm.config = step_llm_config["config"]

    # Act
    builder.add_step(
        template=prompt_template.template,
        output_key="translation",
        required_params=prompt_template.required_params,
        output_json_structure=prompt_template.output_json_structure,
        llm=step_llm_config
    )
    chain = builder.build()

    # Assert
    assert len(chain.steps) == 1
    step = chain.steps[0]
    assert step.output_key == "translation"
    assert step.llm.provider == "google"
    assert step.llm.model == "gemini-1.5"
    assert step.llm.config['temperature'] == 0.5
    assert step.llm.config['max_tokens'] == 1000
    assert step.llm.config['vector_db_provider'] == "mock_provider"

# TODO: Uncomment this after figuring out how to mock streaming.
# @pytest.mark.asyncio
# async def test_llm_chain_stream_async_full_response(llm_chain, mock_llm):
#     # Arrange
#     initial_input = {"name": "Frank"}
    
#     # Create an instance of AsyncIteratorMock with the desired outputs
#     stream_mock = AsyncIteratorMock("Hello, ", "Frank!")
    
#     # Define an async function that returns the stream_mock when awaited
#     async def mock_stream_input_async(*args, **kwargs):
#         return stream_mock
    
#     # Configure the mock_llm's stream_input_async to use the side effect
#     mock_llm.stream_input_async.side_effect = mock_stream_input_async

#     # Act
#     responses = []
#     async for output in llm_chain.stream_async(initial_input):
#         responses.append(output)

#     # Assert
#     assert len(responses) == 3
#     assert responses[0] == {"name": "Frank"}
#     assert responses[1] == {"greeting": {"greeting": "Hello, "}}
#     assert responses[2] == {"greeting": {"greeting": "Frank!"}}

# TODO: Uncomment this after figuring out how to mock streaming. 
# @pytest.mark.asyncio
# async def test_llm_chain_stream_async_json_partial_then_dict(llm_chain, mock_llm):
#     # Arrange
#     initial_input = {"name": "Grace"}
    
#     # Simulate streaming JSON in parts
#     async_mock_stream = AsyncIteratorMock('{"greeting": "Hello, ', 'Grace!"}')
    
#     # Configure the mock_llm's stream_input_async to return the AsyncIteratorMock instance
#     mock_llm.stream_input_async.return_value = async_mock_stream
    
#     # Act
#     responses = []
#     async for output in llm_chain.stream_async(initial_input):
#         responses.append(output)
    
#     # Assert
#     mock_llm.stream_input_async.assert_awaited_once_with("Hello, Grace!", parse_json=True)
#     assert responses == [{"greeting": "Hello, "}, {"greeting": "Grace!"}]

def test_llm_chain_builder_extract_placeholders():
    # Arrange
    builder = LLMChainBuilder()
    template = "Generate a report for {{year}} in {{region}}."

    # Act
    placeholders = builder._extract_placeholders(template)

    # Assert
    assert placeholders == {"year": Any, "region": Any}

def test_llm_chain_builder_extract_placeholders_with_quotes():
    # Arrange
    builder = LLMChainBuilder()
    template = '"""This is a quoted {{placeholder}}""" and outside {{another}}.'

    # Act
    placeholders = builder._extract_placeholders(template)

    # Assert
    assert placeholders == {"another": Any}

@pytest.mark.asyncio
async def test_llm_chain_invalid_json_fallback(llm_chain, mock_llm):
    # Arrange
    text = "Incomplete JSON: {\"key\": \"value\""
    mock_llm.send_input_async.return_value = {"extracted_json": {"key": "value"}}

    # Act
    result = await llm_chain._extract_json(text)

    # Assert
    assert result == {"key": "value"}

@pytest.mark.asyncio
async def test_llm_chain_execute_multiple_steps(mock_llm):
    # Arrange
    prompt_template1 = PromptTemplate(
        template="Set value to {{value}}",
        required_params={"value": int},
        output_json_structure={"set_value": int}
    )
    step1 = LLMChainStep(prompt_template1, "set_value")
    
    prompt_template2 = PromptTemplate(
        template="Increment value to {{value}}",
        required_params={"value": int},
        output_json_structure={"incremented_value": int}
    )
    step2 = LLMChainStep(prompt_template2, "incremented_value")
    
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[step1, step2],
        use_memory=False
    )
    initial_input = {"value": 10}
    mock_llm.send_input_async.side_effect = [
        {"set_value": 10},
        {"incremented_value": 11}
    ]

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    assert result == {"value": 10, "set_value": {"set_value": 10}, "incremented_value": {"incremented_value": 11}}
    assert mock_llm.send_input_async.await_count == 2

def test_llm_chain_builder_missing_required_params(mock_llm):
    # Arrange
    builder = LLMChainBuilder()
    template = "Only {{existing}} placeholder."
    
    # Act
    builder.add_step(
        template=template,
        output_key="response",
        required_params={"existing": str},
        output_json_structure={"response": str}
    )
    chain = builder.build()

    # Mock the LLM response
    chain.default_llm.send_input_async.return_value = {"response": "OK"}

    # Act
    result = asyncio.run(chain.execute_async({"existing": "data"}))
    
    # Assert
    assert result == {"existing": "data", "response": {"response": "OK"}}

@pytest.mark.asyncio
async def test_llm_chain_validate_complex_types():
    # Arrange
    prompt_template = PromptTemplate(
        template="Process {{data}} and {{info}}",
        required_params={"data": List[int], "info": Dict[str, str]},
        output_json_structure={"processed_data": List[int], "processed_info": Dict[str, str]}
    )
    step = LLMChainStep(prompt_template, "processed_results")
    chain = LLMChain(
        default_llm=AsyncMock(),
        steps=[step],
        use_memory=False
    )
    initial_input = {
        "data": [1, 2, 3],
        "info": {"key1": "value1", "key2": "value2"}
    }
    chain.default_llm.send_input_async.return_value = {
        "processed_results": {
            "processed_data": [1, 2, 3],
            "processed_info": {"key1": "value1", "key2": "value2"}
        }
    }

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    chain.default_llm.send_input_async.assert_awaited_once()
    assert result == {
        "data": [1, 2, 3],
        "info": {"key1": "value1", "key2": "value2"},
        "processed_results": {
            "processed_results": {
                "processed_data": [1, 2, 3],
                "processed_info": {"key1": "value1", "key2": "value2"}
            }
        }
    }

@pytest.mark.asyncio
async def test_llm_chain_builder_step_with_no_required_params():
    # Arrange
    builder = LLMChainBuilder()
    template = "Say hello!"
    prompt_template = PromptTemplate(
        template=template,
        required_params={},  # No required params
        output_json_structure={"response": str}
    )
    
    with patch('llmcore.chain.PromptTemplate') as MockPromptTemplate:
        MockPromptTemplate.return_value = prompt_template
        mock_llm = AsyncMock()
        mock_llm.send_input_async.return_value = {"response": "Hello!"}
        
        # Act
        builder.add_step(
            template=template,
            output_key="response",
            required_params={},  # Explicitly no required params
            output_json_structure={"response": str}
        )
        chain = builder.build()
        result = await chain.execute_async({})  # No initial input needed

    # Assert
    assert result == {"response": {"response": "Hello!"}}

@pytest.mark.asyncio
async def test_llm_chain_validate_input_dict_with_any_value():
    # Arrange
    prompt_template = PromptTemplate(
        template="Process {{data}}",
        required_params={"data": Dict[str, Any]},
        output_json_structure={"processed_data": Dict[str, Any]}
    )
    step = LLMChainStep(prompt_template, "processed_data")
    chain = LLMChain(
        default_llm=AsyncMock(),
        steps=[step],
        use_memory=False
    )
    initial_input = {"data": {"key1": "value1", "key2": 2}}

    # Mock the LLM response
    chain.default_llm.send_input_async.return_value = {"processed_data": {"key1": "value1", "key2": 2}}

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    chain.default_llm.send_input_async.assert_awaited_once()
    assert result == {"data": {"key1": "value1", "key2": 2}, "processed_data": {"processed_data": {"key1": "value1", "key2": 2}}}

@pytest.mark.asyncio
async def test_llm_chain_execute_async_empty_steps(mock_llm):
    # Arrange
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[],
        use_memory=False
    )
    initial_input = {"key": "value"}

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    mock_llm.send_input_async.assert_not_called()
    assert result == {"key": "value"}

@pytest.mark.asyncio
async def test_llm_chain_execute_async_step_with_no_output_json_structure(mock_llm):
    # Arrange
    prompt_template = PromptTemplate(
        template="Just echo {{message}}",
        required_params={"message": str},
        output_json_structure=None  # No JSON structure
    )
    step = LLMChainStep(prompt_template, "echo")
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[step],
        use_memory=False
    )
    initial_input = {"message": "Hello!"}
    mock_llm.send_input_async.return_value = "Echo: Hello!"

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    mock_llm.send_input_async.assert_awaited_once()
    assert result == {"message": "Hello!", "echo": "Echo: Hello!"}

@pytest.mark.asyncio
async def test_llm_chain_execute_async_step_with_custom_llm():
    # Arrange
    default_llm = AsyncMock()
    custom_llm = AsyncMock()
    prompt_template_default = PromptTemplate(
        template="Default LLM says {{text}}",
        required_params={"text": str},
        output_json_structure={"response_default": str}
    )
    step_default = LLMChainStep(prompt_template_default, "response_default")
    
    prompt_template_custom = PromptTemplate(
        template="Custom LLM processes {{text}}",
        required_params={"text": str},
        output_json_structure={"response_custom": str}
    )
    step_custom = LLMChainStep(prompt_template_custom, "response_custom", llm=custom_llm)
    
    chain = LLMChain(
        default_llm=default_llm,
        steps=[step_default, step_custom],
        use_memory=False
    )
    initial_input = {"text": "Test"}
    
    default_llm.send_input_async.return_value = {"response_default": "Default response"}
    custom_llm.send_input_async.return_value = {"response_custom": "Custom response"}

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    assert result == {
        "text": "Test",
        "response_default": {"response_default": "Default response"},
        "response_custom": {"response_custom": "Custom response"}
    }

@pytest.mark.asyncio
async def test_llm_chain_execute_async_nested_steps(mock_llm):
    # Arrange
    prompt_template1 = PromptTemplate(
        template="Initialize with {{init}}",
        required_params={"init": int},
        output_json_structure={"init_value": int}
    )
    step1 = LLMChainStep(prompt_template1, "init_value")
    
    prompt_template2 = PromptTemplate(
        template="Double the value to {{init_value.init_value}}",
        required_params={"init_value": Dict[str, int]},
        output_json_structure={"double_value": int}
    )
    step2 = LLMChainStep(prompt_template2, "double_value")
    
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[step1, step2],
        use_memory=False
    )
    initial_input = {"init": 4}
    mock_llm.send_input_async.side_effect = [
        {"init_value": 4},
        {"double_value": 8}
    ]

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    assert result == {"init": 4, "init_value": {"init_value": 4}, "double_value": {"double_value": 8}}
    assert mock_llm.send_input_async.await_count == 2

@pytest.mark.asyncio
async def test_llm_chain_builder_add_step_without_output_json_structure(mock_llm):
    # Arrange
    builder = LLMChainBuilder()
    prompt_template = PromptTemplate(
        template="Print {{message}}",
        required_params={"message": str},
        output_json_structure=None
    )
    
    with patch('llmcore.chain.PromptTemplate') as MockPromptTemplate:
        MockPromptTemplate.return_value = prompt_template
        mock_llm = AsyncMock()
        mock_llm.send_input_async.return_value = "Printed message"

        # Act
        builder.add_step(
            template=prompt_template.template,
            output_key="print_output",
            required_params=prompt_template.required_params,
            output_json_structure=None,
            llm=mock_llm
        )
        chain = builder.build()
        result = await chain.execute_async({"message": "Hello World"})

    # Assert
    assert result == {"message": "Hello World", "print_output": "Printed message"}

@pytest.mark.asyncio
async def test_llm_chain_invalid_placeholder_format(mock_llm):
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        prompt_template = PromptTemplate(
            template="Missing closing brace {{name",
            required_params={"name": str},
            output_json_structure={"response": str}
        )
        step = LLMChainStep(prompt_template, "response")
        chain = LLMChain(
            default_llm=mock_llm,
            steps=[step],
            use_memory=False
        )
        initial_input = {"name": "Henry"}
        await chain.execute_async(initial_input)

@pytest.mark.asyncio
async def test_llm_chain_execute_async_large_context(mock_llm):
    # Arrange
    prompt_template = PromptTemplate(
        template="Summarize the following data: {{data}}",
        required_params={"data": str},
        output_json_structure={"summary": str}
    )
    step = LLMChainStep(prompt_template, "summary")
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[step],
        use_memory=False
    )
    large_data = "A" * 10000  # Simulate large input
    initial_input = {"data": large_data}
    mock_llm.send_input_async.return_value = {"summary": "Summary of large data."}

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    assert result == {"data": large_data, "summary": {"summary": "Summary of large data."}} 

@pytest.mark.asyncio
async def test_llm_chain_execute_async_with_extra_parameters(mock_llm):
    # Arrange
    prompt_template = PromptTemplate(
        template="Greet {{name}} with {{greeting}}",
        required_params={"name": str, "greeting": str},
        output_json_structure={"full_greeting": str}
    )
    step = LLMChainStep(prompt_template, "full_greeting")
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[step],
        use_memory=False
    )
    initial_input = {"name": "Ivy", "greeting": "Good morning!", "extra_param": "should be ignored"}
    mock_llm.send_input_async.return_value = {"full_greeting": "Good morning! Ivy"}

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    assert result == {
        "name": "Ivy",
        "greeting": "Good morning!",
        "extra_param": "should be ignored",
        "full_greeting": {"full_greeting": "Good morning! Ivy"}
    }

@pytest.mark.asyncio
async def test_llm_chain_execute_async_with_empty_string_parameter(mock_llm):
    # Arrange
    prompt_template = PromptTemplate(
        template="Echo {{message}}",
        required_params={"message": str},
        output_json_structure={"echo": str}
    )
    step = LLMChainStep(prompt_template, "echo")
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[step],
        use_memory=False
    )
    initial_input = {"message": ""}
    mock_llm.send_input_async.return_value = {"echo": ""}

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    assert result == {"message": "", "echo": {"echo": ""}}

@pytest.mark.asyncio
async def test_llm_chain_execute_async_invalid_json_in_response(llm_chain, mock_llm):
    # Arrange
    initial_input = {"name": "Jack"}
    mock_llm.send_input_async.return_value = "Invalid JSON response"

    # Act
    result = await llm_chain.execute_async(initial_input)

    # Assert
    # Since output_json_structure is {"greeting": str}, attempt to set greeting as string should work
    assert result == {"name": "Jack", "greeting": "Invalid JSON response"}

@pytest.mark.asyncio
async def test_llm_chain_execute_async_multiple_steps_with_memory(mock_llm):
    # Arrange
    prompt_template1 = PromptTemplate(
        template="Store {{data}}",
        required_params={"data": str},
        output_json_structure={"stored_data": str}
    )
    step1 = LLMChainStep(prompt_template1, "stored_data")
    
    prompt_template2 = PromptTemplate(
        template="Retrieve stored data: {{stored_data.stored_data}}",
        required_params={"stored_data": Dict[str, str]},
        output_json_structure={"retrieved_data": str}
    )
    step2 = LLMChainStep(prompt_template2, "retrieved_data")
    
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[step1, step2],
        use_memory=True
    )
    initial_input = {"data": "Sample Data"}
    mock_llm.send_input_with_memory.side_effect = [
        ({"stored_data": "Sample Data"}),
        ({"retrieved_data": "Sample Data"})
    ]

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    assert result == ({"data": "Sample Data", "stored_data": {"stored_data": "Sample Data"}, "retrieved_data": {"retrieved_data": "Sample Data"}})
    assert mock_llm.send_input_with_memory.await_count == 2

@pytest.mark.asyncio
async def test_llm_chain_execute_async_with_partial_json_response(llm_chain, mock_llm):
    # Arrange
    initial_input = {"name": "Karen"}
    # Simulate partial JSON response that needs to be parsed
    mock_llm.send_input_async.return_value = {"greeting": "Hello, Karen!"}

    # Act
    result = await llm_chain.execute_async(initial_input)

    # Assert
    assert result == {"name": "Karen", "greeting": {"greeting": "Hello, Karen!"}}

# TODO: Uncomment this after figuring out how to mock streaming. 
# @pytest.mark.asyncio
# async def test_llm_chain_stream_async_no_json_yields_full_response(llm_chain, mock_llm):
#     # Arrange
#     initial_input = {"name": "Laura"}
    
#     # Create an instance of AsyncIteratorMock with the full response
#     async_mock_stream = AsyncIteratorMock("Hello, Laura!")
    
#     # Configure the mock_llm's stream_input_async to return the AsyncIteratorMock instance
#     mock_llm.stream_input_async.side_effect = async_mock_stream
    
#     # Act
#     responses = []
#     async for output in llm_chain.stream_async(initial_input):
#         responses.append(output)
    
#     # Assert
#     mock_llm.stream_input_async.assert_awaited_once_with("Hello, Laura!", parse_json=True)
#     assert responses == [{"greeting": {"greeting": "Hello, Laura!"}}]

@pytest.mark.asyncio
async def test_llm_chain_execute_async_with_extra_large_input(mock_llm):
    # Arrange
    prompt_template = PromptTemplate(
        template="Analyze the following text: {{text}}",
        required_params={"text": str},
        output_json_structure={"analysis": str}
    )
    step = LLMChainStep(prompt_template, "analysis")
    chain = LLMChain(
        default_llm=mock_llm,
        steps=[step],
        use_memory=False
    )
    large_text = "A" * 5000
    initial_input = {"text": large_text}
    mock_llm.send_input_async.return_value = {"analysis": "Analysis complete."}

    # Act
    result = await chain.execute_async(initial_input)

    # Assert
    assert result == {"text": large_text, "analysis": {"analysis": "Analysis complete."}}

@pytest.mark.asyncio
async def test_llm_chain_execute_async_invalid_json_fallback_no_json_found(llm_chain, mock_llm):
    # Arrange
    text = "No JSON here."
    mock_llm.send_input_async.return_value = {"extracted_json": {}}

    # Act
    result = await llm_chain._extract_json(text)

    # Assert
    mock_llm.send_input_async.assert_awaited_once()
    assert result == {}