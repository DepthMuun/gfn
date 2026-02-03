# Contributing to Manifold

Thank you for your interest in contributing to Manifold! This document provides comprehensive guidelines and detailed instructions for contributing to the Geodesic Flow Networks project. Whether you are fixing bugs, adding new features, improving documentation, or proposing innovative ideas to extend the geometric sequence modeling capabilities, your contributions are warmly welcomed and deeply appreciated. This guide will walk you through the entire contribution process, from setting up your development environment to having your changes merged into the main repository.

## 1. Getting Started

### 1.1 Prerequisites and Environment Requirements

Before you begin contributing to Manifold, it is essential to ensure that your development environment meets all the necessary requirements and is properly configured. Manifold is built on PyTorch and relies on several scientific computing libraries, so having the correct versions installed is crucial for successful development and testing.

The following tools and dependencies must be installed on your system before you can contribute to Manifold. Python version 3.10 or higher is required, as the project utilizes modern Python features including structural pattern matching and improved type hint syntax. Git is essential for version control operations including cloning repositories, creating branches, and managing commits. PyTorch version 2.0 or higher is the core deep learning framework upon which Manifold is built, and version compatibility is important for certain features to function correctly. A CUDA-compatible GPU is strongly recommended for development and testing, as the project involves numerical computations that benefit significantly from GPU acceleration. A LaTeX distribution is necessary for building mathematical documentation that contains properly rendered equations and symbols.

You can verify your Python version by running `python --version` in your terminal. For PyTorch installation with CUDA support, visit the official PyTorch website to get the appropriate installation command for your system configuration. GPU availability can be verified using the `nvidia-smi` command on Linux systems or by running a simple PyTorch tensor computation on the GPU.

### 1.2 Setting Up Your Development Environment

Establishing a proper development environment is the first and most important step in contributing to Manifold. The following procedure will guide you through forking the repository, cloning it to your local machine, and configuring all necessary development dependencies.

First, navigate to the main Manifold repository page on GitHub and click the "Fork" button in the upper-right corner of the page. This creates your personal copy of the repository under your GitHub account, allowing you to make changes without affecting the main project. After forking is complete, clone your forked repository to your local machine using the git clone command with the URL of your fork. Navigate into the repository directory and install the package in editable mode along with all development dependencies.

```bash
# Clone your forked repository
git clone https://github.com/Manifold-Laboratory/manifold.git
cd manifold

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks for automated quality checks
pre-commit install
```

The editable installation (`-e` flag) allows you to modify the source code and have those changes immediately reflected without reinstalling the package. The development dependencies include testing frameworks, documentation building tools, linting utilities, and other essential development tools. Pre-commit hooks are automated scripts that run before each commit to ensure your code meets the project's quality standards, including formatting checks, linting, and basic sanity tests.

After installation, verify that your environment is correctly set up by running a simple test. Import the Manifold module and verify that you can create an instance of the model without errors. If you encounter any import errors or dependency issues, consult the [Troubleshooting](docs/troubleshooting.md) documentation for guidance on resolving common setup problems.

### 1.3 Understanding the Project Structure

Manifold follows a carefully organized modular architecture designed to separate concerns, facilitate code reuse, and make the system extensible. Before making significant contributions, take the time to understand how the project is structured and how different components interact with each other.

The `src/` directory contains the main implementation source code, organized into logical submodules that correspond to different components of the system. This includes modules for the core manifold model, integrators for numerical simulation, optimizers for Riemannian gradient descent, and utility functions for data processing and evaluation. Each submodule contains clear documentation and type annotations to facilitate understanding and extension.

The `tests/` directory contains comprehensive test suites organized by component type. Unit tests verify individual components in isolation, ensuring that functions and classes behave correctly with various inputs. Integration tests verify that multiple components work together correctly, testing the interaction between different parts of the system. End-to-end tests verify complete workflows from input to output, ensuring the system functions correctly as a complete pipeline.

The `docs/` directory holds all project documentation including getting started guides, tutorials, mathematical foundations, architecture references, API documentation, benchmarking results, and troubleshooting guides. This documentation is maintained alongside the code and should be updated whenever changes affect user-facing behavior or introduce new concepts.

The `notebooks/` directory contains Jupyter notebooks for experimentation, visualization, and demonstration purposes. These notebooks can be useful for exploring the model's behavior, visualizing latent trajectories, and creating examples for documentation.

The `examples/` directory contains complete example scripts demonstrating how to use Manifold for various tasks. These examples serve as templates for users and as reference implementations for common use cases.

## 2. Coding Standards and Guidelines

### 2.1 Code Style and Formatting

Manifold maintains strict coding standards to ensure code quality, readability, and consistency across the entire codebase. All contributions must adhere to these standards, and the pre-commit hooks will verify compliance before allowing commits. Following these standards helps ensure that code is maintainable, readable, and consistent across all contributions regardless of the author.

Python code must follow PEP 8 style guidelines with specific modifications tailored to this project. Use four spaces for indentation rather than tabs, as this ensures consistent rendering across different editors and platforms. Maximum line length is 120 characters, which accommodates complex mathematical expressions and type annotations while still maintaining reasonable line lengths for readability.

All public functions, classes, and modules must have comprehensive docstrings following the NumPy docstring format. This format provides a consistent structure for documentation and is compatible with automatic documentation generation tools. Docstrings should include descriptions of the function's purpose, parameters, return values, and any exceptions that may be raised.

Naming conventions in Manifold follow established Python practices with some domain-specific additions. Function and variable names should be descriptive and use lowercase with underscores for readability, such as `compute_geodesic_step` or `initial_momentum`. Class names should use CamelCase convention, such as `ManifoldModel` or `LeapfrogIntegrator`. Constants should be written in UPPERCASE with underscores separating words, such as `DEFAULT_INTEGRATOR_STEPS` or `MAX_GRADIENT_NORM`.

Avoid single-letter variable names except for mathematical indices and temporary loop variables where the context is clear. When working with mathematical objects, prefer descriptive names like `position`, `velocity`, and `momentum` over abstract names like `x`, `v`, and `p`, though single-letter names may be used in local contexts where the mathematical meaning is immediately apparent.

### 2.2 Type Hints and Type Safety

Type hints are required for all function signatures and class methods in Manifold. This requirement ensures type safety, provides better IDE support for development, and enables automatic documentation generation. Type hints also serve as inline documentation, making it easier for reviewers and future maintainers to understand the expected inputs and outputs of each function.

Use the `typing` module for complex types including optional types, union types, and generic types. For example, function parameters that may be None should be typed as `Optional[torch.Tensor]` rather than relying on implicit None handling. Lists of tensors should be typed as `List[torch.Tensor]` or `Sequence[torch.Tensor]` depending on whether mutability is required.

Consider using `TypeAlias` for type aliases that improve code readability, especially for complex types that appear frequently in the codebase. For example, if your module frequently works with pairs of position and velocity tensors, you might define a type alias like `StateVector = Tuple[torch.Tensor, torch.Tensor]` and use this alias throughout the code.

```python
from typing import Optional, List, TypeAlias, Tuple
import torch

StateVector: TypeAlias = Tuple[torch.Tensor, torch.Tensor]

class ManifoldModel:
    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[StateVector] = None
    ) -> Tuple[torch.Tensor, StateVector]:
        """Forward pass through the manifold model.
        
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            state: Optional initial state (position, velocity) tuple
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            state: Final state (position, velocity) tuple
        """
        ...
```

### 2.3 Mathematical Notation and Documentation

Given Manifold's foundation in geometric mechanics and Riemannian geometry, proper mathematical notation is essential for code comments, docstrings, and documentation. When documenting mathematical concepts, following consistent notation helps bridge the gap between the theoretical foundations and their computational implementations.

Variables should be clearly distinguished by their mathematical meaning and physical interpretation. Position vectors should be denoted as $x_t$, representing the semantic state at time step $t$. Velocity vectors should be denoted as $v_t$ or $\dot{x}_t$, representing the rate of change of position and encoding contextual momentum. Momentum vectors should be denoted as $p_t$, representing the conjugate momentum in the Hamiltonian formulation.

Christoffel symbols, which define the connection coefficients on the Riemannian manifold, should be denoted with appropriate indices as $\Gamma^k_{ij}$. The upper index indicates the contravariant component, while the two lower indices indicate the covariant components of the connection. When implementing Christoffel symbol computations in code, maintain this index structure in the tensor shapes to preserve the mathematical meaning.

Hamiltonian functions should be written as $\mathcal{H}(x, v)$, representing the total energy of the system as a function of position and velocity. The Hamiltonian governs the time evolution of the system through Hamilton's equations, and its conservation is crucial for the numerical stability of the model over long sequences.

Complex equations spanning multiple lines should use LaTeX notation for proper rendering in documentation. Inline mathematics should be enclosed in single dollar signs, while display mathematics should use double dollar signs with appropriate line breaks for readability.

```python
def compute_christoffel_symbols(g: torch.Tensor, dg: torch.Tensor) -> torch.Tensor:
    """Compute Christoffel symbols of the second kind.
    
    The Christoffel symbols are computed from the metric tensor g and
    its derivative dg using the formula:
    
    Γ^k_ij = (1/2) g^kl (∂g_li/∂x^j + ∂g_lj/∂x^i - ∂g_ij/∂x^l)
    
    where g^kl is the inverse metric tensor.
    
    Args:
        g: Metric tensor of shape (..., d, d)
        dg: Derivative of metric tensor (..., d, d, d)
        
    Returns:
        Christoffel symbols Γ^k_ij of shape (..., d, d, d)
    """
    ...
```

## 3. Testing Requirements

### 3.1 Test Coverage and Quality Standards

Manifold maintains comprehensive test coverage to ensure code quality, prevent regressions, and verify correctness across different use cases and scenarios. All new features must include appropriate test coverage, and bug fixes should include regression tests that verify the fix works correctly and prevent the bug from reoccurring in the future.

The test suite in Manifold is organized into several categories that serve different purposes in the development workflow. Unit tests verify individual components in isolation, checking that functions and classes behave correctly with various inputs including edge cases and boundary conditions. These tests should be fast to execute and focus on testing specific functionality without external dependencies.

Integration tests verify that multiple components work together correctly, testing the interaction between different parts of the system. These tests ensure that when components are combined, they produce the expected results and that interfaces between components are correctly implemented.

End-to-end tests verify complete workflows from input to output, ensuring the system works correctly as a whole. These tests simulate real-world usage scenarios and verify that the complete pipeline functions correctly, including data loading, model execution, and output processing.

Benchmark tests measure performance metrics including inference speed, memory usage, and numerical accuracy. These tests ensure that optimizations do not degrade performance and that the system meets the performance requirements for production use.

Before submitting any contribution, ensure that all existing tests pass with your changes and that your changes do not introduce any regressions in functionality that was previously working correctly. The project maintainers will not merge contributions that reduce test coverage or introduce test failures.

### 3.2 Running the Test Suite

The test suite can be executed using pytest, which is the standard testing framework for Python projects. Several commands are available to run different subsets of the test suite depending on your needs during development.

To run all tests with coverage reporting, use the following command. The coverage report will show which lines of code are exercised by the tests and identify areas that may need additional testing. Aiming for high coverage helps ensure that the codebase is thoroughly tested.

```bash
# Run all tests with coverage report
pytest --cov=src --cov-report=term-missing --cov-report=html

# Run with verbose output for debugging
pytest -v --tb=short

# Run tests matching a specific pattern
pytest -k "test_manifold" -v

# Run specific test file
pytest tests/units/test_model.py -v

# Run with detailed output for a specific test
pytest tests/units/test_model.py::TestManifoldModel::test_forward -v --tb=long
```

During active development, you may want to run only the tests relevant to your current changes to save time. Use the `-k` flag to filter tests by name, or specify the path to specific test files or directories. When you believe your changes are complete, run the full test suite to ensure no regressions have been introduced.

### 3.3 Writing High-Quality Tests

When writing tests for Manifold, following these guidelines will ensure high-quality test coverage that effectively validates the code and provides value for future maintenance:

Each test should focus on a single behavior or functionality. Avoid testing multiple unrelated things in a single test function, as this makes it harder to diagnose failures and reduces the clarity of the test suite. When a test fails, the test name should immediately indicate what specific functionality is not working correctly.

Use descriptive test names that clearly indicate what is being tested and what the expected behavior is. A good test name follows the pattern `test_{function}_{scenario}_{expected_result}` or similar structure that conveys the test's purpose without requiring the reader to examine the test body.

```python
class TestLeapfrogIntegrator:
    """Test suite for the Leapfrog symplectic integrator."""
    
    def test_energy_conservation(self, symplectic_system):
        """Verify that energy is conserved over many integration steps."""
        integrator = LeapfrogIntegrator(stepsize=0.01)
        initial_energy = symplectic_system.hamiltonian()
        
        for _ in range(1000):
            integrator.step(symplectic_system)
        
        final_energy = symplectic_system.hamiltonian()
        energy_change = abs(final_energy - initial_energy)
        
        # Energy should be conserved to numerical precision
        assert energy_change < 1e-6, (
            f"Energy not conserved: changed by {energy_change}"
        )
```

Use pytest fixtures for setup and teardown operations to promote code reuse and ensure consistent test setup across multiple tests. Fixtures should be defined in conftest.py files or using the `@pytest.fixture` decorator and should be named descriptively to indicate what they provide.

For mathematical operations, include tests that verify numerical accuracy against known analytical solutions or reference implementations. Consider edge cases and boundary conditions, testing behavior at extreme values and with invalid inputs. Mock external dependencies when appropriate to isolate the code under test and ensure tests run quickly and reliably.

## 4. Pull Request Process

### 4.1 Preparing Your Changes for Review

Before submitting a pull request, it is essential to thoroughly prepare your changes to ensure they are ready for review and can be merged successfully. Taking the time to prepare properly will result in faster review times and higher quality contributions.

Begin by updating your local branch with the latest changes from the main repository to avoid merge conflicts. Rebase your changes onto the latest main branch to ensure your contribution can be cleanly merged without conflicts. Run the full test suite one final time to verify that all tests pass with your changes, including any new tests you have written.

Review your changes carefully, checking for any unintended modifications or leftover debug code. Ensure that your changes only include the modifications necessary to implement the feature or fix the bug you are addressing. Remove any commented-out code, temporary debugging statements, or unnecessary files.

Commit your changes with clear, descriptive commit messages that explain what was changed and why. Each commit should represent a logical unit of change that can be understood independently. For complex changes, consider breaking them into multiple commits that each address a specific aspect of the feature.

### 4.2 Creating an Effective Pull Request

When creating a pull request on GitHub, following these guidelines will help reviewers understand your changes and provide effective feedback:

Use a descriptive title that summarizes the changes in the pull request. The title should be specific enough to identify the contribution but concise enough to be easily understood at a glance. Avoid generic titles like "Fix bug" or "Add feature" in favor of titles that describe what was changed.

The pull request description should thoroughly explain what was changed, why these changes were made, and how they affect the codebase. Include any relevant context such as the problem being solved, the approach taken, and any trade-offs made. If the pull request addresses an existing issue, include a reference to that issue using GitHub's auto-linking syntax.

If your pull request introduces new features or changes existing behavior, you must update the documentation accordingly. This includes updating docstrings for changed functions and classes, adding examples for new features, and modifying relevant documentation files in the docs/ directory. Documentation is as important as code in Manifold, and pull requests without appropriate documentation updates will not be merged.

```markdown
## Summary

This pull request implements [brief description of changes].

## Changes Made

- [Detail of change 1]
- [Detail of change 2]
- [Detail of change 3]

## Motivation

[Explain why these changes are necessary or beneficial]

## Testing

- [Describe how changes were tested]
- [Note any new tests added]

## Checklist

- [ ] Tests pass
- [ ] Documentation updated
- [ ] Type hints added/updated
- [ ] Code follows style guidelines
```

### 4.3 Responding to Review Feedback

The review process is an iterative dialogue between you and the project maintainers. Being responsive and constructive during review will lead to better contributions and a smoother merging process.

Reviewers will examine your code for correctness, style compliance, documentation quality, and test coverage. Feedback will typically be provided within a few business days, though complex contributions may require more time for thorough review. Be prepared to address feedback by making additional changes to your pull request.

When implementing changes requested by reviewers, clearly indicate what changes you have made. You can either add new commits to your branch or amend existing commits to incorporate the feedback. The final history will be cleaned up when the pull request is merged, so focus on ensuring the changes are correct rather than maintaining a clean commit history during review.

If you disagree with feedback or suggestions from reviewers, explain your reasoning clearly and respectfully. There may be valid technical reasons for different approaches, and open discussion often leads to better solutions. However, be open to the expertise of reviewers who may have context you are not aware of.

Once all feedback has been addressed and the review is approved, a maintainer will merge your pull request into the main branch. Your contributions will be acknowledged in the project's commit history and will be included in the next release notes.

## 5. Documentation Standards

### 5.1 Documentation Requirements for Contributions

All contributions to Manifold must include appropriate documentation updates. This requirement reflects the project's commitment to providing comprehensive and accessible documentation for all users. Documentation is considered an integral part of any contribution, and pull requests without appropriate documentation updates will not be merged.

When adding new functionality, you must update docstrings for all new public functions and classes. These docstrings should clearly describe what the function or class does, what parameters it accepts, what it returns, and any exceptions it may raise. Examples should be included where they help clarify usage.

When modifying existing functionality, update the relevant documentation to reflect any changes in behavior, parameters, or usage. This includes updating docstrings, modifying existing examples, and adding new examples if the change affects how the functionality is used.

When adding new concepts or changing the architecture, update the relevant documentation files in the docs/ directory. This may include updating the architecture documentation, adding new sections to the mathematical foundations, or creating new documentation files for significant new features.

### 5.2 Writing Clear Documentation

Documentation in Manifold should be written in clear, professional English with proper grammar and punctuation. Technical terms should be used correctly and consistently throughout the documentation, following the terminology established in the mathematical foundations and existing documentation.

Use Markdown formatting consistently throughout all documentation files. Section headings should follow a logical hierarchy, starting with H2 (##) for major sections and using H3 (###) and H4 (####) for subsections as needed. Code blocks should use appropriate language tags for syntax highlighting.

Include cross-references to related documentation when relevant. This helps users navigate the documentation and find additional information about related topics. Use relative links to other documentation files in the docs/ directory.

When introducing new concepts, provide examples and context to aid understanding. Examples should be self-contained and runnable, demonstrating the concept in a practical context. Include comments in examples to explain what each part does and why.

```markdown
## Computing Geodesic Steps

The geodesic equation describes how a particle moves along the shortest path
on a curved manifold. In Manifold, we approximate this motion using discrete
integration schemes that preserve the geometric properties of the flow.

### Using the Leapfrog Integrator

The leapfrog integrator provides a good balance between accuracy and
computational efficiency for geodesic computation:

```python
from gfn.integrators import LeapfrogIntegrator

integrator = LeapfrogIntegrator(stepsize=0.01)
position, velocity = integrator.step(position, velocity, forces)
```

For more details on numerical integration schemes, see
[Integrators](architecture.md#integrators).
```

## 6. Community Guidelines

### 6.1 Professional Communication

Contributors are expected to communicate respectfully and professionally in all project interactions. This includes pull request discussions, issue comments, code review responses, and any other project communication channels including community forums and direct messages.

When engaging with other contributors and maintainers, assume positive intent and approach discussions with a collaborative mindset. Disagreements are natural in software development, but they should be handled professionally and respectfully. Focus on the technical merits of different approaches rather than personal characteristics.

Ask clarifying questions when something is unclear rather than making assumptions. Provide context and background information when requesting changes or suggesting alternatives. Be constructive in your feedback, explaining not just what should be changed but why the change would be beneficial.

### 6.2 Recognition and Attribution

All contributions to Manifold are recognized and appreciated, regardless of their size or scope. Contributors are acknowledged in release notes and, with their permission, in the project's contributor list that appears in documentation and on the project website.

Significant contributions may be highlighted in blog posts or other communications from the Manifold Laboratory. This includes major feature implementations, significant bug fixes, substantial documentation improvements, and other contributions that have a meaningful impact on the project.

## 7. Additional Resources

### 7.1 Documentation Reference

The following documentation resources are available to help you understand Manifold and contribute effectively:

The [Getting Started](docs/getting-started.md) guide provides a comprehensive introduction to installing and configuring Manifold, including environment setup, basic configuration options, and quick-start examples for users who are new to the project. The [Tutorial](docs/tutorial.md) offers a step-by-step example of building a complete sequence modeling project using Manifold, demonstrating best practices and common patterns for effective usage.

The [Mathematical Foundations](docs/mathematical-foundations.md) document provides an in-depth exploration of the geometric mechanics principles underlying Manifold, covering Hamiltonian dynamics, symplectic flows, Riemannian geometry, and the physical interpretation of sequence modeling as a dynamical system. The [Architecture](docs/architecture.md) guide provides detailed information about system components, data flow, and design decisions, essential for understanding how Manifold processes sequences internally and for making modifications to the core implementation.

The [API Reference](docs/API.md) contains complete documentation for all public interfaces including classes, functions, and modules, with parameter descriptions, return values, and usage examples for integration into production systems. The [Benchmarking](docs/benchmarking.md) document provides detailed documentation of performance evaluations and comparison studies, including methodology, metrics, datasets, and results comparing Manifold against Transformers, Mamba, and other state space models.

The [Troubleshooting](docs/troubleshooting.md) guide provides solutions to common issues, guides for resolving numerical instability problems, performance optimization tips, and answers to frequently asked questions. This guide is essential for debugging problems and successfully deploying Manifold in production environments.

### 7.2 Getting Help

If you have questions or need assistance with contributing to Manifold, several resources are available to help you:

First, check existing issues and discussions for similar questions, as your question may have already been answered. Review the documentation thoroughly before asking new questions, as the answer may already be available in one of the documentation files. When asking questions, provide relevant context including what you have tried, what you expected to happen, and what actually happened.

For questions about the project, contributing guidelines, or technical implementation, open a new issue with the "question" label. This creates a public record of the question and answer that can help other contributors with similar questions in the future.

For urgent issues or sensitive topics, you may contact the project maintainers directly through GitHub. However, public channels are preferred for most questions, as they allow the community to benefit from the discussion and provide diverse perspectives on the answer.

### 7.3 Acknowledgments

Thank you for considering contributing to Manifold! Your contributions help advance the field of geometric sequence modeling and make powerful tools accessible to researchers and practitioners worldwide. Every contribution, whether it is a bug fix, a new feature, documentation improvement, or a feature suggestion, helps make Manifold a better project for everyone.

The Manifold Laboratory team and the community of contributors look forward to your contributions and are committed to supporting you throughout the contribution process. Welcome to the community!

---


*Last updated: February 2026*
*For questions about this guide, please open an issue on GitHub*
