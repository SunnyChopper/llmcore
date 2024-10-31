import pytest
from typing import List, Dict, Set, Union, Optional, Any
from llmcore.prompt import PromptTemplate

def test_prompt_template_creation():
    template = "Hello, {{name}}!"
    required_params = {"name": str}
    prompt_template = PromptTemplate(template, required_params)
    assert prompt_template.template == template
    assert prompt_template.required_params == required_params
    assert prompt_template.placeholders == {"name"}

def test_prompt_template_validation():
    template = "Hello, {{name}}!"
    required_params = {"name": str}
    prompt_template = PromptTemplate(template, required_params)
    with pytest.raises(ValueError):
        prompt_template._validate_inputs({})
    with pytest.raises(TypeError):
        prompt_template._validate_inputs({"name": 123})
    prompt_template._validate_inputs({"name": "World"})

def test_convert_types_basic_types():
    template = "test"
    required_params = {}
    output_structure = {
        "string_field": str,
        "int_field": int,
        "bool_field": bool
    }
    
    prompt = PromptTemplate(template, required_params, output_structure)
    result = prompt.output_json_structure
    
    assert result == {
        "string_field": "str",
        "int_field": "int",
        "bool_field": "bool"
    }

def test_convert_types_list():
    template = "test"
    required_params = {}
    output_structure = {
        "string_list": List[str],
        "int_list": List[int],
        "nested_list": List[List[str]]
    }
    
    prompt = PromptTemplate(template, required_params, output_structure)
    result = prompt.output_json_structure
    
    assert result == {
        "string_list": "list[str]",
        "int_list": "list[int]",
        "nested_list": "list[list[str]]"
    }

def test_convert_types_dict():
    template = "test"
    required_params = {}
    output_structure = {
        "string_dict": Dict[str, str],
        "int_dict": Dict[str, int],
        "any_dict": Dict[str, Any]
    }
    
    prompt = PromptTemplate(template, required_params, output_structure)
    result = prompt.output_json_structure
    
    assert result == {
        "string_dict": "dict[str, str]",
        "int_dict": "dict[str, int]",
        "any_dict": "dict[str, typing.Any]"
    }

def test_convert_types_set():
    template = "test"
    required_params = {}
    output_structure = {
        "string_set": Set[str],
        "int_set": Set[int]
    }
    
    prompt = PromptTemplate(template, required_params, output_structure)
    result = prompt.output_json_structure
    
    assert result == {
        "string_set": "set[str]",
        "int_set": "set[int]"
    }

def test_convert_types_union():
    template = "test"
    required_params = {}
    output_structure = {
        "string_or_int": Union[str, int],
        "multi_union": Union[str, int, bool]
    }
    
    prompt = PromptTemplate(template, required_params, output_structure)
    result = prompt.output_json_structure
    
    assert result == {
        "string_or_int": "Union[str, int]",
        "multi_union": "Union[str, int, bool]"
    }

def test_convert_types_optional():
    template = "test"
    required_params = {}
    output_structure = {
        "optional_string": Optional[str],
        "optional_list": Optional[List[int]]
    }
    
    prompt = PromptTemplate(template, required_params, output_structure)
    result = prompt.output_json_structure
    
    assert result == {
        "optional_string": "Union[str, NoneType]",
        "optional_list": "Union[list[int], NoneType]"
    }

def test_convert_types_nested_structures():
    template = "test"
    required_params = {}
    output_structure = {
        "complex_field": Dict[str, List[Union[str, int]]],
        "nested_optional": Optional[Dict[str, List[str]]],
        "mixed_types": Dict[str, Union[List[int], Dict[str, Any]]]
    }
    
    prompt = PromptTemplate(template, required_params, output_structure)
    result = prompt.output_json_structure
    
    assert result == {
        "complex_field": "dict[str, list[Union[str, int]]]",
        "nested_optional": "Union[dict[str, list[str]], NoneType]",
        "mixed_types": "dict[str, Union[list[int], dict[str, typing.Any]]]"
    }
