<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AI Assistant Development Project Instructions

This project focuses on building an AI assistant from scratch without using pre-trained models. When providing code suggestions:

## Code Style & Standards
- Use PyTorch as the primary deep learning framework
- Follow PEP 8 Python style guidelines
- Include comprehensive docstrings for all functions and classes
- Use type hints for better code clarity
- Implement proper error handling and logging

## Project Structure Guidelines
- Organize code by development phases (phase1_fundamentals through phase8_features)
- Keep shared utilities in the `shared/` directory
- Place configuration files in `configs/`
- Store experimental notebooks in `notebooks/`

## Deep Learning Best Practices
- Implement transformer architecture components from scratch
- Use gradient checkpointing for memory efficiency
- Include proper weight initialization
- Implement learning rate scheduling
- Add model checkpointing and resuming capabilities

## Data Processing Guidelines
- Implement efficient data loaders with proper batching
- Include data validation and preprocessing steps
- Use tokenization best practices (BPE/WordPiece)
- Handle large datasets with streaming capabilities

## Training Considerations
- Include proper loss calculation and backward pass
- Implement gradient clipping for stability
- Add comprehensive logging and metrics tracking
- Support distributed training when applicable
- Include evaluation loops and model testing

## Application Development
- Create modular, reusable components
- Implement proper conversation handling
- Include user input validation and sanitization
- Add proper error handling for user interactions
- Design scalable architecture for production use

## Documentation & Testing
- Include clear README files for each phase
- Add comprehensive unit tests
- Document all hyperparameters and configuration options
- Provide example usage for all components

Remember: This project aims to build everything from scratch for educational purposes. Avoid suggesting pre-trained models or shortcuts that bypass the learning objectives.
