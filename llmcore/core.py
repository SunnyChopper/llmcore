from typing import Dict, Any, Union, AsyncGenerator, Generator, List
from abc import ABC, abstractmethod
from urllib.parse import urljoin
import aiohttp
import asyncio
import typing
import json
import os
import re

from llmcore.prompt import Prompt
from llmcore.logger import setup_logger, log
from llmcore.config import LLMConfig
from llmcore.contracts import ConversationTurn
from llmcore.memory import MemoryManager
from llmcore.embeddings import Embeddings

logger = setup_logger(__name__)

class LLMAPIError(Exception):
    pass

class LLMJSONParseError(Exception):
    pass

class LLMPromptError(Exception):
    pass

class LLMNetworkError(Exception):
    pass

class APIEndpoints:
    OPENAI = "https://api.openai.com/v1"
    ANTHROPIC = "https://api.anthropic.com/v1"
    GOOGLE_GEMINI = "https://generativelanguage.googleapis.com/v1beta"

class LLMClientAdapter(ABC):
    @abstractmethod
    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        pass

    @abstractmethod
    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        pass

class APIClientAdapter(LLMClientAdapter):
    def __init__(self, api_key: str, base_url: str, model: str, max_retries: int = 3, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    # NOTE: This is for debugging purposes only. It allows you to copy and paste the
    #       cURL command into your terminal to test the API call in case you're having
    #       trouble getting the request to work.
    def _generate_curl_command(self, url: str, data: Dict, headers: Dict) -> str:
        header_args = ' '.join([f"-H '{k}: {v}'" for k, v in headers.items()])
        data_arg = f"-d '{json.dumps(data)}'"
        return f"curl -X POST {header_args} {data_arg} '{url}'"

    async def _make_request(self, endpoint: str, data: Dict, headers: Dict) -> Dict:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        # curl_command = self._generate_curl_command(url, data, headers)
        # log(logger, "INFO", f"cURL command for debugging:\n{curl_command}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=data, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientResponseError as e:
                error_message = f"API request failed: {str(e)}"
                try:
                    error_content = e.message
                    error_message += f"\nResponse content: {error_content}"
                except:
                    error_message += "\nCouldn't retrieve response content."
                
                log(logger, "ERROR", error_message)
                log(logger, "ERROR", f"Args: {e.args}")
                log(logger, "ERROR", f"Status: {e.status}")
                log(logger, "ERROR", f"Message: {e.message}")
                log(logger, "ERROR", f"Headers: {e.headers}")
                log(logger, "ERROR", f"Request Info: {e.request_info}")
                log(logger, "ERROR", f"URL: {e.request_info.url}")
                log(logger, "ERROR", f"Method: {e.request_info.method}")
                log(logger, "ERROR", f"Headers: {e.request_info.headers}")
                
                if hasattr(e.request_info, 'body'):
                    log(logger, "ERROR", f"Request Body: {e.request_info.body}")
                
                if hasattr(e, 'history'):
                    for i, resp in enumerate(e.history):
                        log(logger, "ERROR", f"Redirect {i + 1}: {resp.status} - {resp.url}")
                
                # Parse the error message to extract more detailed information
                try:
                    error_dict = json.loads(e.message)
                    if 'error' in error_dict and 'message' in error_dict['error']:
                        detailed_error = error_dict['error']['message']
                        log(logger, "ERROR", f"Detailed error: {detailed_error}")
                except json.JSONDecodeError:
                    log(logger, "ERROR", "Failed to parse error message as JSON")
                
                raise LLMAPIError(error_message)
            except aiohttp.ClientError as e:
                error_message = f"Network error occurred: {str(e)}"
                log(logger, "ERROR", error_message)
                raise LLMNetworkError(error_message)

    async def _stream_request(self, endpoint: str, data: Dict, headers: Dict) -> AsyncGenerator[Dict, None]:
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                if self.model.startswith("claude-"):
                    async for chunk in self._stream_anthropic(response):
                        yield chunk
                elif self.model.startswith("gpt-"):
                    async for chunk in self._stream_openai(response):
                        yield chunk
                elif self.model.startswith("gemini-"):
                    async for chunk in self._stream_google(response):
                        yield chunk
                else:
                    async for chunk in self._stream_default(response):
                        yield chunk

    async def _stream_anthropic(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        parsed_json = json.loads(json_str)
                        yield parsed_json
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON

    async def _stream_openai(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(json_str)
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON

    async def _stream_google(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(json_str)
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON

    async def _stream_default(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict, None]:
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON


    @abstractmethod
    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        pass

    @abstractmethod
    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        pass

class OpenAIClientAdapter(APIClientAdapter):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, APIEndpoints.OPENAI, model)

    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p
        }

        if config.response_format:
            data["response_format"] = config.response_format

        response = await self._make_request("/chat/completions", data, headers)
        return response["choices"][0]["message"]["content"]

    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }
        try:
            async for chunk in self._stream_request("/chat/completions", data, headers):
                if "choices" in chunk and chunk["choices"]:
                    if "content" in chunk["choices"][0]["delta"]:
                        yield chunk["choices"][0]["delta"]["content"]
        except Exception as e:
            raise e

class AnthropicClientAdapter(APIClientAdapter):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, APIEndpoints.ANTHROPIC, model)

    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        try:
            response = await self._make_request("/messages", data, headers)
            return response["content"][0]["text"]
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                try:
                    error_content = json.loads(e.message)
                    if error_content.get("type") == "error" and error_content.get("error", {}).get("type") == "invalid_request_error":
                        error_message = error_content["error"]["message"]
                        if "credit balance is too low" in error_message:
                            raise LLMAPIError(f"Anthropic API request failed: Insufficient credits. {error_message}")
                except json.JSONDecodeError as json_err:
                    log(logger, "ERROR", f"Failed to parse error response as JSON: {json_err}")
                    log(logger, "DEBUG", f"Raw error response: {e.message}")
            log(logger, "ERROR", f"Anthropic API error: {e}")
            raise LLMAPIError(f"Anthropic API request failed: {e}")
        except Exception as e:
            log(logger, "ERROR", f"Anthropic API error: {e}")
            raise LLMAPIError(f"Anthropic API request failed: {e}")

    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }

        try:
            async for chunk in self._stream_request("/messages", data, headers):
                if "delta" in chunk and "text" in chunk["delta"]:
                    yield chunk["delta"]["text"]
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                try:
                    error_content = json.loads(e.message)
                    if error_content.get("type") == "error" and error_content.get("error", {}).get("type") == "invalid_request_error":
                        error_message = error_content["error"]["message"]
                        if "credit balance is too low" in error_message:
                            raise LLMAPIError(f"Anthropic API request failed: Insufficient credits. {error_message}")
                except json.JSONDecodeError as json_err:
                    log(logger, "ERROR", f"Failed to parse error response as JSON: {json_err}")
                    log(logger, "DEBUG", f"Raw error response: {e.message}")
            raise LLMAPIError(f"Anthropic API request failed: {e}")
        except Exception as e:
            raise LLMAPIError(f"Anthropic API request failed: {e}")

class GoogleGeminiClientAdapter(APIClientAdapter):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, APIEndpoints.GOOGLE_GEMINI, model)

    async def send_prompt(self, prompt: str, config: LLMConfig) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "topP": config.top_p,
                "maxOutputTokens": config.max_tokens,
            }
        }
        endpoint = f"/models/{self.model}:generateContent?key={self.api_key}"
        response = await self._make_request(endpoint, data, headers)
        return response["candidates"][0]["content"]["parts"][0]["text"]

    async def stream_prompt(self, prompt: str, config: LLMConfig) -> AsyncGenerator[str, None]:
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.temperature,
                "topP": config.top_p,
                "maxOutputTokens": config.max_tokens,
            }
        }
        endpoint = f"/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        async for chunk in self._stream_request(endpoint, data, headers):
            if "candidates" in chunk and chunk["candidates"]:
                content = chunk["candidates"][0]["content"]["parts"][0]["text"]
                if content:
                    yield content

class LLM:
    JSON_ENSURE_RESPONSE = "\n\nPlease ensure your entire response is valid JSON."
    
    def __init__(self, provider: str, model: str, config: LLMConfig = LLMConfig()):
        self.provider = provider
        self.model = model
        self.config = config
        self.client = self.load_model()
        self.embeddings = Embeddings(provider="openai", model="text-embedding-3-small")
        self.memory_manager = MemoryManager(config = config, capacity = 32000)

    def load_model(self):
        api_provider = self.provider.lower() if self.provider != "google" else "gemini"
        api_key = os.environ.get(f"{api_provider.upper()}_API_KEY")
        if not api_key:
            raise ValueError(f"API key for {self.provider} not found in environment variables")

        if self.provider == "openai":
            return OpenAIClientAdapter(api_key, self.model)
        elif self.provider == "anthropic":
            return AnthropicClientAdapter(api_key, self.model)
        elif self.provider == "gemini" or self.provider == "google":
            return GoogleGeminiClientAdapter(api_key, self.model)
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")

    def send_input(self, prompt: Prompt, parse_json: bool = False) -> Union[str, Dict[str, Any]]:
        return asyncio.run(self._send_input_async(prompt, parse_json))

    def stream_input(self, prompt: Prompt, parse_json: bool = False) -> Generator[Union[str, Dict[str, Any]], None, None]:
        async def async_generator():
            async for chunk in self.stream_input_async(prompt, parse_json=parse_json):
                yield chunk

        return self._async_to_sync_generator(async_generator())

    async def send_input_async(self, prompt: Prompt, parse_json: bool = False) -> Union[str, Dict[str, Any]]:
        log(logger, "DEBUG", f"Sending input: {prompt}")
        formatted_prompt = prompt.format()
        return await self._send_input_async(formatted_prompt, parse_json, prompt.template.output_json_structure)

    async def stream_input_async(self, prompt: Prompt, parse_json: bool = False) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        formatted_prompt = prompt.format()
        self._configure_json_mode(prompt.template.output_json_structure is not None)

        try:
            accumulated_json = ""
            async for chunk in self.client.stream_prompt(formatted_prompt, self.config):
                if parse_json:
                    yield chunk  # Stream the raw chunk
                    accumulated_json += chunk
                    try:
                        parsed_json = json.loads(accumulated_json)
                        log(logger, "DEBUG", f"Parsed JSON: {parsed_json}")
                        yield self._extract_fields(parsed_json, prompt.template.output_json_structure)
                        accumulated_json = ""  # Reset after successful parse
                    except json.JSONDecodeError:
                        # Continue accumulating if it's not a complete JSON yet
                        continue
                else:
                    yield chunk

        except aiohttp.ClientError as e:
            log(logger, "ERROR", f"Network error: {str(e)}")
            raise LLMNetworkError(f"Network error occurred while streaming from LLM: {str(e)}")
        except json.JSONDecodeError as e:
            log(logger, "ERROR", f"JSON parse error: {str(e)}")
            raise LLMJSONParseError(f"Failed to parse JSON from LLM response: {str(e)}")
        except Exception as e:
            log(logger, "ERROR", f"Unexpected error: {str(e)}")
            raise LLMAPIError(f"Unexpected error occurred while streaming from LLM: {str(e)}")

    async def send_input_with_history(self, prompt: Prompt, history: List[ConversationTurn], parse_json: bool = False) -> Union[str, Dict[str, Any]]:
        formatted_prompt = prompt.format()
        full_prompt = self._build_prompt_with_history(formatted_prompt, history)
        return await self._send_input_async(full_prompt, parse_json, prompt.template.output_json_structure)

    def _build_prompt_with_history(self, current_prompt: str, history: List[ConversationTurn]) -> str:
        conversation = "\n".join([f"{turn.role}: {turn.content}" for turn in history])
        return f"{conversation}\n\nHuman: {current_prompt}\nAI:"

    async def send_input_with_memory(self, prompt: Prompt, parse_json: bool = False) -> Union[str, Dict[str, Any]]:
        try:
            formatted_prompt = prompt.format()
            log(logger, "INFO", f"Formatted prompt: {formatted_prompt[:50]}...")  # Log first 50 chars
            
            try:
                relevant_memories = await self.memory_manager.get_relevant_memories(formatted_prompt)
                log(logger, "INFO", f"Retrieved {len(relevant_memories)} relevant memories")
            except Exception as e:
                log(logger, "ERROR", f"Error retrieving relevant memories: {str(e)}")
                relevant_memories = []
            
            memory_context = "\n".join([f"Memory: {mem['content'][:30]}..." for mem in relevant_memories])  # Truncate each memory
            full_prompt = f"{memory_context}\n\nHuman: {formatted_prompt}\nAI:"
            log(logger, "INFO", f"Full prompt length: {len(full_prompt)} characters")
            log(logger, "INFO", f"Full prompt: {full_prompt}")
            
            try:
                response = await self._send_input_async(full_prompt, parse_json, prompt.template.output_json_structure)
                log(logger, "DEBUG", f"Received response of type: {type(response).__name__}")
            except Exception as e:
                log(logger, "ERROR", f"Error sending input to LLM: {str(e)}")
                raise

            try:
                vector = await self.embeddings.embed_async(formatted_prompt)
            except Exception as e:
                log(logger, "ERROR", f"Error embedding prompt: {str(e)}")
                vector = None
            
            try:
                # Add the new interaction to memory
                await self.memory_manager.add_memory({
                    "content": formatted_prompt,
                    "response": str(response)[:50] + "...",
                    "vector": vector or formatted_prompt
                })
                log(logger, "DEBUG", "Added new interaction to memory")
            except Exception as e:
                log(logger, "ERROR", f"Error adding interaction to memory: {str(e)}")
            
            return response
        except Exception as e:
            log(logger, "ERROR", f"Unexpected error in send_input_with_memory: {str(e)}")
            raise

    async def _send_input_async(self, formatted_prompt: str, parse_json: bool, output_json_structure: Dict[str, Any] = None) -> Union[str, Dict[str, Any]]:
        log(logger, "DEBUG", f"Sending input to {self.provider} {self.model}")
        self._configure_json_mode(output_json_structure is not None)

        try:
            log(logger, "DEBUG", f"Prompt: {formatted_prompt}")
            response = await self.client.send_prompt(formatted_prompt, self.config)
            log(logger, "DEBUG", f"Raw response: {response}")

            if parse_json:
                parsed_response = self._parse_json_response(response, output_json_structure)
                log(logger, "DEBUG", f"Parsed response: {parsed_response}")
                return parsed_response
            return response
        except json.JSONDecodeError as e:
            log(logger, "ERROR", f"JSON parse error: {str(e)}")
            raise LLMJSONParseError(f"Failed to parse JSON from the LLM response: {str(e)}")
        except LLMJSONParseError as e:
            log(logger, "ERROR", str(e))
            raise
        except aiohttp.ClientError as e:
            log(logger, "ERROR", f"Network error: {str(e)}")
            raise LLMNetworkError(f"Network error occurred while sending prompt to LLM: {str(e)}")
        except Exception as e:
            log(logger, "ERROR", f"Unexpected error: {str(e)}")
            raise LLMAPIError(f"Unexpected error occurred while sending prompt to LLM: {str(e)}")

    def _parse_json_response(self, response: str, expected_structure: Dict[str, Any]) -> Dict[str, Any]:
        log(logger, "DEBUG", f"Parsing JSON response: {response}")
        try:
            parsed_json = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from the response if it's not already in JSON format
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(1))
                    log(logger, "DEBUG", f"Parsed JSON: {parsed_json}")
                except json.JSONDecodeError:
                    log(logger, "ERROR", "Failed to parse JSON from the LLM response")
                    raise LLMJSONParseError("Failed to parse JSON from the LLM response")
            else:
                log(logger, "ERROR", "Failed to parse JSON from the LLM response")
                raise LLMJSONParseError("Failed to parse JSON from the LLM response")

        return self._extract_fields(parsed_json, expected_structure)

    def _extract_fields(self, parsed_json: Dict[str, Any], expected_structure: Dict[str, Any]) -> Dict[str, Any]:
        log(logger, "DEBUG", f"Extracting fields: {expected_structure}")
        result = {}
        for key, expected_type in expected_structure.items():
            if key not in parsed_json:
                log(logger, "ERROR", f"Missing expected field: {key}")
                raise LLMJSONParseError(f"Missing expected field: {key}")
            
            value = parsed_json[key]
            log(logger, "DEBUG", f"Value: {value}")
            if isinstance(expected_type, str):
                result[key] = self._validate_type(value, expected_type, key)
            elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                valid_types = [t for t in expected_type.__args__ if not isinstance(t, type(typing.Any))]
                if not any(self._is_valid_type(value, t) for t in valid_types):
                    log(logger, "ERROR", f"Field {key} should be one of {valid_types}")
                    raise LLMJSONParseError(f"Field {key} should be one of {valid_types}")
                result[key] = value
            else:
                result[key] = self._validate_type(value, expected_type, key)
        
        return result

    def _is_valid_type(self, value, expected_type):
        if hasattr(expected_type, '__origin__'):
            if expected_type.__origin__ is dict:
                return isinstance(value, dict)
            elif expected_type.__origin__ is list:
                return isinstance(value, list)
        return isinstance(value, expected_type)

    def _validate_type(
        self, value: Any, expected_type: Union[str, type, Dict[str, Any]], field_name: str
    ) -> Any:
        if isinstance(expected_type, dict):
            if not isinstance(value, dict):
                log(logger, "ERROR", f"Field {field_name} should be a dictionary")
                raise LLMJSONParseError(f"Field {field_name} should be a dictionary")
            return self._extract_fields(value, expected_type)

        if isinstance(expected_type, str):
            if expected_type == 'int':
                if not isinstance(value, (int, float)):
                    log(logger, "ERROR", f"Field {field_name} should be an integer or float")
                    raise LLMJSONParseError(f"Field {field_name} should be an integer or float")
                return int(value)
            elif expected_type == 'float':
                if not isinstance(value, (int, float)):
                    log(logger, "ERROR", f"Field {field_name} should be a float")
                    raise LLMJSONParseError(f"Field {field_name} should be a float")
                return float(value)
            elif expected_type == 'str':
                if not isinstance(value, str):
                    log(logger, "ERROR", f"Field {field_name} should be a string")
                    raise LLMJSONParseError(f"Field {field_name} should be a string")
                return value
            elif expected_type == 'bool':
                if not isinstance(value, bool):
                    log(logger, "ERROR", f"Field {field_name} should be a boolean")
                    raise LLMJSONParseError(f"Field {field_name} should be a boolean")
                return value
            elif expected_type.startswith('dict['):
                if not isinstance(value, dict):
                    log(logger, "ERROR", f"Field {field_name} should be a dictionary")
                    raise LLMJSONParseError(f"Field {field_name} should be a dictionary")
                return value
            elif expected_type.startswith('list['):
                if not isinstance(value, list):
                    log(logger, "ERROR", f"Field {field_name} should be a list")
                    raise LLMJSONParseError(f"Field {field_name} should be a list")
                return value
            elif expected_type.startswith('Union['):
                # Handle Union types
                union_types = expected_type[6:-1].split(',')
                for union_type in union_types:
                    try:
                        return self._validate_type(value, union_type.strip(), field_name)
                    except LLMJSONParseError:
                        continue
                log(logger, "ERROR", f"Field {field_name} does not match any type in {expected_type}")
                raise LLMJSONParseError(f"Field {field_name} does not match any type in {expected_type}")
            else:
                log(logger, "ERROR", f"Unsupported type for field {field_name}: {expected_type}")
                raise LLMJSONParseError(f"Unsupported type for field {field_name}: {expected_type}")

        elif isinstance(expected_type, type):
            if not isinstance(value, expected_type):
                log(logger, "ERROR", f"Field {field_name} should be of type {expected_type.__name__}")
                raise LLMJSONParseError(f"Field {field_name} should be of type {expected_type.__name__}")
            return value

        elif hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
            # Handle typing.Union types
            for union_type in expected_type.__args__:
                try:
                    return self._validate_type(value, union_type, field_name)
                except LLMJSONParseError:
                    continue
            log(logger, "ERROR", f"Field {field_name} does not match any type in {expected_type}")
            raise LLMJSONParseError(f"Field {field_name} does not match any type in {expected_type}")

        else:
            log(logger, "ERROR", f"Unsupported type for field {field_name}: {expected_type}")
            raise LLMJSONParseError(f"Unsupported type for field {field_name}: {expected_type}")
        
    def _configure_json_mode(self, json_response: bool):
        if json_response:
            if self.provider == "openai" and "gpt-4" in self.model:
                self.config.response_format = {"type": "json_object"}
            elif self.provider in ["anthropic", "google"]:
                self.config.json_response = True
                if "claude-3" in self.model or "gemini" in self.model:
                    self.config.json_instruction = self.JSON_ENSURE_RESPONSE
        else:
            self.config.response_format = None

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                log(logger, "ERROR", f"Invalid configuration option: {key}")
                raise ValueError(f"Invalid configuration option: {key}")

    @staticmethod
    def _async_to_sync_generator(async_gen):
        agen = async_gen.__aiter__()
        loop = asyncio.get_event_loop()
        try:
            while True:
                yield loop.run_until_complete(agen.__anext__())
        except StopAsyncIteration:
            pass