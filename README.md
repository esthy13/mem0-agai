# Exercise Repository

Welcome to the course exercise repository! This README will guide you through setting up your development environment and submitting your group work.

## Getting Started

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Set Up Your Environment with `uv`

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

Install the project in editable mode:

```bash
uv pip install -e .
```

This allows you to make changes to the code that are immediately reflected without reinstalling.

### 3. Update Dependencies

When adding new dependencies for your submission, update the `pyproject.toml` file:

```toml
[project]
dependencies = [
    "your-package>=1.0.0",
    # Add your required packages here
]
```

After updating `pyproject.toml`, sync your environment:

```bash
uv pip install -e .
```

### 4. Create `.env` File
Create a `.env` file in the root directory of the project with the following content:
```
LLM_API_KEY=sk-1234
LLM_BASE_URL=http://localhost:4000
```
Replace `sk-1234` with your actual API key and `http://localhost:4000` with base URL of the API, described in moodle.
*Do not* commit this file to the repository.

## Workflow

### Working with GitLab Issues and Merge Requests
1. Exercises are handled via GitLab Issues and Merge Requests (MRs).

2. **Merge Request:** Create a Merge Request (MR). Assign yourself to the MR.       
3. **Create a Branch:** Create a new branch for your work from the Merge Request page from the `exercise_1` branch.
4. **Naming Convention:** Both your **branch name** and the **folder** you create within `exercise_1/` MUST follow this pattern:
    `{exercise_short}_group_{character}`
    *   `{exercise_short}` is provided in the GitLab issue description.
    *   `group_{character}` represents your assigned group (e.g., `group_a`).
    *   Example: `react_group_a`

## Code Quality Standards

We expect high-quality, professional code submissions. Please adhere to the following:

### Code Style
- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Use meaningful variable and function names
- Keep functions focused and concise (single responsibility principle)
- Maintain consistent indentation and formatting

### Documentation
- **All functions, classes, and modules must be documented** using [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include clear descriptions, parameter types, return types, and examples where appropriate

Example:
```python
def calculate_average(numbers: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        numbers: A list of numerical values.

    Returns:
        The average of the input numbers.

    Raises:
        ValueError: If the list is empty.

    Examples:
        >>> calculate_average([1, 2, 3, 4, 5])
        3.0
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
```

### Testing
- Write unit tests for your code
- Ensure all tests pass before submitting your MR
- Aim for meaningful test coverage

## Documentation Website

This repository automatically generates documentation from your code docstrings:

- **Documentation Tool**: [pdoc](https://pdoc.dev/)
- **Trigger**: GitLab CI pipeline on the `main` branch
- **Format**: Google-style docstrings (as shown above)

Your properly formatted docstrings will be automatically compiled into a browsable documentation website when code is merged to `main`. This makes it essential to document your code thoroughly!

## Need Help?
Via Moodle or florian.schroeder@techfak.de