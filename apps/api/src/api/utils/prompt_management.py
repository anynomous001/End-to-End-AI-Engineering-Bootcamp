import yaml
from pathlib import Path
from jinja2 import Template


# Path to the prompts directory (relative to this file)
PROMPTS_DIR = Path(__file__).parent.parent / "agents" / "prompts"


def prompt_template_config(yaml_file: str | Path, prompt_key: str) -> Template:
    """
    Load a Jinja2 Template from a YAML prompt registry file.

    Args:
        yaml_file: Path to the YAML file containing prompt definitions.
        prompt_key: Key under 'prompts' in the YAML to load.

    Returns:
        A compiled Jinja2 Template.
    """
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    template_content = config["prompts"][prompt_key]

    template = Template(template_content)

    return template


def build_prompt_jinja(preprocessed_context: str, question: str) -> str:
    """
    Render the retrieval-generation prompt using the YAML prompt registry.

    Args:
        preprocessed_context: Formatted product context string.
        question: User's question.

    Returns:
        Rendered prompt string ready for LLM consumption.
    """
    yaml_path = PROMPTS_DIR / "retrieval_generation.yaml"

    template = prompt_template_config(yaml_path, "retrieval_generation")

    rendered_prompt = template.render(
        preprocessed_context=preprocessed_context,
        question=question,
    )

    return rendered_prompt
