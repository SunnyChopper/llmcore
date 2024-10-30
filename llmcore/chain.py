# System imports
from typing import List, Dict, Any, Optional, Union, get_origin, get_args, Iterator, AsyncIterator
import asyncio
import json
import re

# LLMCore imports
from llmcore.core import LLM, LLMConfig
from llmcore.prompt import PromptTemplate

class LLMChainError(Exception):
    pass

class LLMChainStep:
    def __init__(self, prompt_template: PromptTemplate, output_key: str, llm: Optional[LLM] = None):
        self.prompt_template = prompt_template
        self.output_key = output_key
        self.llm = llm
        self.required_params = prompt_template.required_params

class LLMChain:
    def __init__(self, default_llm: LLM, steps: List[LLMChainStep], use_memory: bool = False):
        self.default_llm = default_llm
        self.steps = steps
        self.context = {}
        self.use_memory = use_memory

    def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.execute_async(initial_input))

    async def execute_async(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        self.context = initial_input.copy()

        for i, step in enumerate(self.steps):
            try:
                self._validate_input(step, self.context)
            except Exception as e:
                raise LLMChainError(f"Input validation failed for step {i+1}: {str(e)}")

            try:
                prompt_context = {
                    **self.context,
                    "previous_steps": {k: v for k, v in self.context.items() if k not in initial_input}
                }
                prompt = step.prompt_template.create_prompt(**prompt_context)
            except Exception as e:
                raise LLMChainError(f"Prompt creation failed for step {i+1}: {str(e)}")

            parse_json = step.prompt_template.output_json_structure is not None
            llm_to_use = step.llm or self.default_llm

            try:
                if self.use_memory:
                    response = await llm_to_use.send_input_with_memory(prompt, parse_json=parse_json)
                else:
                    response = await llm_to_use.send_input_async(prompt, parse_json=parse_json)
            except Exception as e:
                raise LLMChainError(f"LLM request failed for step {i+1}: {str(e)}")

            try:
                self.context[step.output_key] = response
            except Exception as e:
                raise LLMChainError(f"Error processing LLM response for step {i+1}: {str(e)}")

        return self.context
    
    def stream(self, initial_input: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        return asyncio.run(self.stream_async(initial_input))

    async def stream_async(self, initial_input: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        self.context = initial_input.copy()

        for i, step in enumerate(self.steps):
            try:
                self._validate_input(step, self.context)

                prompt_context = {
                    **self.context,
                    "previous_steps": {k: v for k, v in self.context.items() if k not in initial_input}
                }

                prompt = step.prompt_template.create_prompt(**prompt_context)
                parse_json = step.prompt_template.output_json_structure is not None
                llm_to_use = step.llm or self.default_llm

                full_response = ""
                async for chunk in llm_to_use.stream_input_async(prompt, parse_json=parse_json):
                    if isinstance(chunk, dict):
                        # Extract the relevant value from the chunk based on the output_key
                        extracted_value = chunk.get(step.output_key)
                        yield {step.output_key: extracted_value}
                    else:
                        full_response += chunk

                extracted_json = await self._extract_json(full_response)
                self.context[step.output_key] = extracted_json
            except Exception as e:
                raise LLMChainError(f"Error in step {i+1}: {str(e)}")

        yield self.context

    async def _extract_json(self, text: Union[str, Dict[str, Any]]) -> Union[Dict[str, Any], str]:
        # If the input is already a dictionary, return it
        if isinstance(text, dict):
            return text

        # If the input is a string, try to parse it as JSON
        if isinstance(text, str):
            # First, try to extract JSON from the entire text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Then, try to extract JSON from code blocks
            code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            code_block_match = re.search(code_block_pattern, text)
            if code_block_match:
                json_str = code_block_match.group(1)
                try:
                    # Unescape newlines and quotes
                    unescaped_json = json_str.replace('\\\\', '\\').replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                    parsed_json = json.loads(unescaped_json)
                    if (isinstance(parsed_json, dict) and parsed_json) or (isinstance(parsed_json, list) and parsed_json):
                        return parsed_json
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            # If that fails, try to find JSON within the text
            json_pattern = r'(?s)\{(?:[^{}]|(?:\{[^{}]*\})*)*\}|\[(?:[^\[\]]|(?:\[[^\[\]]*\])*)*\]'
            json_matches = re.finditer(json_pattern, text)
            valid_jsons = []
            for json_match in json_matches:
                try:
                    json_str = json_match.group()
                    # Unescape newlines and quotes
                    unescaped_json = json_str.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
                    parsed_json = json.loads(unescaped_json)
                    if (isinstance(parsed_json, dict) and parsed_json) or (isinstance(parsed_json, list) and parsed_json):
                        valid_jsons.append((len(json_str), parsed_json))
                except json.JSONDecodeError:
                    pass

            if valid_jsons:
                # Return the largest valid JSON object found
                return max(valid_jsons, key=lambda x: x[0])[1]

        # If all else fails, use a fast LLM to extract JSON
        return await self._extract_json_with_llm(text)

    async def _extract_json_with_llm(self, text: str) -> Union[Dict[str, Any], str]:
        fast_llm = LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0.3, max_tokens=2048))
        prompt = PromptTemplate(
            "Extract any JSON from the following text. Remove any escape characters and polish the JSON. If no JSON is found, return an empty JSON object with an \"extracted_json\" key that is an empty JSON object \{\"extracted_json\": \{\}\}.\n\nText: {{text}}",
            required_params={"text": str},
            output_json_structure={"extracted_json": Union[Dict[str, Any], List[Any]]}
        )
        result = await fast_llm.send_input_async(prompt.create_prompt(text=text), parse_json=True)
        return result.get("extracted_json", {})

    def _validate_input(self, step, context):
        for param, param_type in step.required_params.items():
            if param not in context:
                raise ValueError(f"Missing required parameter: {param}")

            # Fixed: Use 'is not Any' to avoid isinstance() with Any
            if param_type is not Any:
                value = context[param]
                origin_type = get_origin(param_type) or param_type

                if origin_type in (List, list):
                    if not isinstance(value, list):
                        raise TypeError(f"Parameter {param} should be a list")
                    if value and get_args(param_type):
                        item_type = get_args(param_type)[0]
                        if not all(isinstance(item, item_type) for item in value):
                            raise TypeError(f"All items in {param} should be of type {item_type}")
                elif origin_type in (Dict, dict):
                    if not isinstance(value, dict):
                        raise TypeError(f"Parameter {param} should be a dict")
                    if get_args(param_type):
                        key_type, value_type = get_args(param_type)
                        if key_type is str and value_type is Any:
                            # Special case for Dict[str, Any]
                            if not all(isinstance(k, str) for k in value.keys()):
                                raise TypeError(f"All keys in {param} should be of type str")
                        elif not all(isinstance(k, key_type) and (value_type is Any or isinstance(v, value_type)) for k, v in value.items()):
                            raise TypeError(f"All items in {param} should be of type {key_type}: {value_type}")
                elif origin_type is None:
                    # Handle cases where origin_type could not be determined
                    if not isinstance(value, param_type):
                        raise TypeError(f"Parameter {param} must be of type {param_type}")
                else:
                    if not isinstance(value, origin_type):
                        raise TypeError(f"Parameter {param} must be of type {param_type}")

class LLMChainBuilder:
    def __init__(self, default_llm: LLM = None, use_memory: bool = False):
        self.default_llm = default_llm if default_llm else LLM(provider="openai", model="gpt-4o-mini", config=LLMConfig(temperature=0.9, max_tokens=4096))
        self.steps = []
        self.use_memory = use_memory

    def add_step(self, 
                 template: str, 
                 output_key: str, 
                 required_params: Optional[Dict[str, type]] = None,
                 output_json_structure: Optional[Dict[str, Any]] = None,
                 llm: Optional[Union[LLM, Dict[str, Any]]] = None) -> 'LLMChainBuilder':
        if required_params is None:
            required_params = self._extract_placeholders(template)
        
        prompt_template = PromptTemplate(template, required_params, output_json_structure)
        
        if isinstance(llm, dict):
            llm = LLM(**llm)
        
        self.steps.append(LLMChainStep(prompt_template, output_key, llm))
        return self

    def _extract_placeholders(self, template: str) -> Dict[str, type]:
        # Use negative lookahead to exclude placeholders within triple quotes
        pattern = r'{{(\w+)}}(?!(?:[^"]*"[^"]*")*[^"]*"[^"]*$)'
        placeholders = set(re.findall(pattern, template))
        return {placeholder: Any for placeholder in placeholders}

    def build(self) -> LLMChain:
        return LLMChain(self.default_llm, self.steps, self.use_memory)