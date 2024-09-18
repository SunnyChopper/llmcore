import re
from typing import Dict, Any, Optional, List, Union, get_origin, get_args
import json

class PromptTemplate:
    def __init__(self, template: str, required_params: Dict[str, type], output_json_structure: Optional[Dict[str, Any]] = None):
        self.template = template
        self.required_params = required_params
        self.output_json_structure = self._convert_types(output_json_structure) if output_json_structure else None
        self.placeholders = self._extract_placeholders()

    def _convert_types(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        def convert(item):
            if isinstance(item, type):
                return item.__name__
            elif isinstance(item, list):
                return [convert(i) for i in item]
            elif isinstance(item, dict):
                return {k: convert(v) for k, v in item.items()}
            elif hasattr(item, '__origin__'):  # This handles Union and other generic types
                origin = item.__origin__
                args = item.__args__
                if origin is Union:
                    return f"Union[{', '.join(convert(arg) for arg in args)}]"
                return f"{origin.__name__}[{', '.join(convert(arg) for arg in args)}]"
            else:
                return str(item)

        return {k: convert(v) for k, v in structure.items()}

    def _extract_placeholders(self) -> set:
        return set(re.findall(r'{{(\w+)}}', self.template))

    def _validate_inputs(self, kwargs: Dict[str, Any]):
        for key, expected_type in self.required_params.items():
            if key not in kwargs:
                raise ValueError(f"Missing required parameter: {key}")
            
            value = kwargs[key]

            if expected_type is Any:
                continue

            origin_type = get_origin(expected_type) or expected_type

            if origin_type in (List, list):
                if not isinstance(value, list):
                    raise TypeError(f"Parameter {key} should be a list")
                if value and get_args(expected_type):
                    item_type = get_args(expected_type)[0]
                    if not all(isinstance(item, item_type) for item in value):
                        raise TypeError(f"All items in {key} should be of type {item_type}")
            elif origin_type in (Dict, dict):
                if not isinstance(value, dict):
                    raise TypeError(f"Parameter {key} should be a dict")
                if get_args(expected_type):
                    key_type, value_type = get_args(expected_type)
                    if not all(isinstance(k, key_type) and isinstance(v, value_type) for k, v in value.items()):
                        raise TypeError(f"All items in {key} should be of type {key_type}: {value_type}")
            elif not isinstance(value, origin_type):
                raise TypeError(f"Parameter {key} should be of type {expected_type}")

    def create_prompt(self, **kwargs) -> 'Prompt':
        self._validate_inputs(kwargs)
        return Prompt(self, **kwargs)

class Prompt:
    def __init__(self, template: PromptTemplate, **kwargs):
        self.template = template
        self.values = kwargs

    def _sanitize_input(self, value: Any) -> str:
        sanitized = str(value)
        sanitized = re.sub(r'[<>]', '', sanitized)
        return sanitized

    def format(self) -> str:
        sanitized_kwargs = {}
        for k, v in self.values.items():
            if isinstance(v, dict):
                sanitized_kwargs[k] = {sk: self._sanitize_input(sv) for sk, sv in v.items()}
            else:
                sanitized_kwargs[k] = self._sanitize_input(v)

        print(f"Sanitized kwargs: {sanitized_kwargs}")
        
        def replace_placeholder(match):
            placeholder = match.group(1)
            start = match.start()
            end = match.end()
            
            # Check if the placeholder is within triple quotes
            triple_quote_before = self.template.template.rfind('"""', 0, start)
            triple_quote_after = self.template.template.find('"""', end)
            
            # Check if the placeholder is within single or double quotes
            single_quote_before = self.template.template.rfind("'", 0, start)
            single_quote_after = self.template.template.find("'", end)
            double_quote_before = self.template.template.rfind('"', 0, start)
            double_quote_after = self.template.template.find('"', end)
            
            # If within any of these quote types, don't replace
            if (triple_quote_before != -1 and triple_quote_after != -1) or \
                (single_quote_before != -1 and single_quote_after != -1) or \
                (double_quote_before != -1 and double_quote_after != -1):
                return match.group(0)
            
            # Handle nested placeholders
            parts = placeholder.split('.')
            value = sanitized_kwargs
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return match.group(0)  # Placeholder not found, return original
            return str(value)
        
        formatted_prompt = re.sub(r'{{([\w.]+)}}', replace_placeholder, self.template.template)
        
        if self.template.output_json_structure:
            json_instruction = "\n\nPlease provide your response in the following JSON format and ensure that the JSON is properly enclosed within triple backticks. Do not include any other text or formatting. Be very careful with the formatting of the JSON, as it may break the rest of the application if not done correctly! Finally, ensure that the JSON produced is not the escaped version of the JSON structure, but the actual JSON object:"
            json_structure = json.dumps(self.template.output_json_structure, indent=2)
            formatted_prompt += f"{json_instruction}\n```json\n{json_structure}\n```"
        
        return formatted_prompt