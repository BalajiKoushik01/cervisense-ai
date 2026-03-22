# Contributing to CerviSense-AI

First off, thank you for considering contributing to CerviSense-AI! It's people like you that make CerviSense-AI such a great project.

## Code of Conduct
By participating in this project, you are expected to uphold our code of conduct. We are dedicated to providing a harassment-free experience for everyone.

## Industrial Research Standards

As an industrial-grade research repository, all contributions must adhere to the following standards:

### 1. Code Quality & Linting
- Ensure your code passes PyLint formatting rules. All Pull Requests will be checked against standard line limits (100 columns).
- **Cognitive Complexity**: We mandate strict adherence to SonarQube's cognitive complexity rules. Function cognitive complexity must be less than `15`. If your logic has extensive branching, you must extract blocks into helper functions.
- Run `pylint --disable=C0114,C0115,C0116` against your modified files before submitting a PR.

### 2. Deep Learning Checkpoints & Environment
- Never push `.pth` or `.pt` model weights to the repository. They must be downloaded externally or ignored via `.gitignore` in the `checkpoints/` directory.
- Avoid introducing arbitrary dependencies to `requirements.txt`. If you add a package, confirm it installs correctly across Linux and Windows environments (e.g. avoid packages reliant strictly on native extensions that frequently break Windows).
- Ensure new neural layers correctly implement `torch.amp.autocast` integration so they run seamlessly in mixed precision.

### 3. Pull Request Process
1. Fork the repository and create your feature branch: `git checkout -b feature/amazing-feature`
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. Submit a Pull Request outlining the problem you're solving, the architectural decision taken, and screenshots of training curves/XAI logs if your change alters mathematical operations.

## Raising an Issue
If you do not have a PR but found a reproducible bug, please file an issue with:
- OS & Hardware setup
- Expected vs Actual behavior
- Traceback logs
