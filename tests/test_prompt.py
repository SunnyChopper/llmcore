import pytest
from llmkit.prompt import Prompt, PromptTemplate

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