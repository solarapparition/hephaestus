"""Functions for extracting text from the LLM's output."""

import re


class ExtractionError(Exception):
    """Raised when an extraction fails."""


def extract_blocks(
    text: str, starting_wrapper: str, ending_wrapper: str
) -> list[str] | None:
    """Extracts specially formatted blocks of text from the LLM's output. `block_type` corresponds to a label for a markdown code block such as `yaml` or `python`."""

    pattern = r"{starting_wrapper}\n(.*?){ending_wrapper}".format(  # pylint:disable=consider-using-f-string
        starting_wrapper=starting_wrapper, ending_wrapper=ending_wrapper
    )
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    extracted_strings = [match.strip() for match in matches]
    return extracted_strings
