
# UnleashedLLM

**Democratizing Access to Uncensored AI Language Models**

A sophisticated command-line interface tool for downloading, managing, and interacting with powerful AI models completely offline.

## Overview

UnleashedLLM is a comprehensive Python application that provides seamless access to uncensored AI language models. The tool enables users to download, manage, and interact with various AI models through an intuitive command-line interface, ensuring complete privacy and offline functionality.

## Features


### Core Functionality
- **Model Management**: Download and manage AI models from a curated registry
- **Interactive Chat**: Engage in real-time conversations with AI models
- **Single Inference**: Perform one-off text generation tasks
- **Offline Operation**: Complete functionality without internet dependency after initial setup

### Advanced Capabilities
- **Paginated Model Browser**: Browse available models with detailed specifications
- **Progress Tracking**: Real-time download progress with speed and ETA indicators
- **System Diagnostics**: Comprehensive system requirement checking
- **Status Dashboard**: Monitor storage usage and model availability
- **Conversation History**: Maintain context across chat sessions

### Supported Models

| Model | Size | Category | Context Length | Capabilities |
|-------|------|----------|----------------|--------------|
| Phi-2 Uncensored | 1.6GB | Lightweight | 2,048 tokens | Text generation, Chat |
| Llama2 7B Uncensored | 4.1GB | Medium | 4,096 tokens | Text generation, Chat, Coding |
| CodeLlama 13B | 7.3GB | Medium | 16,384 tokens | Coding, Text generation |
| Mixtral 8x7B | 26.9GB | Large | 32,768 tokens | Text generation, Chat, Reasoning |
| OpenChat 7B | 4.2GB | Medium | 8,192 tokens | Text generation, Chat, Reasoning |

## System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Linux (Nix-based environment)
- **Memory**: 8GB RAM minimum (16GB recommended for larger models)
- **Storage**: Variable (1.6GB - 27GB per model)
- **CPU**: Multi-core processor recommended

## Installation

### Prerequisites

The application automatically handles dependency installation, including:
- `llama-cpp-python`: Core inference engine
- Standard Python libraries for networking and file management

### Setup

1. Clone or download the project files
2. Ensure Python 3.11+ is installed
3. Run the application - dependencies will be installed automatically

```bash
python3 main.py
```

## Usage

### Command Line Interface

Launch the application to access the main menu:

```bash
python3 main.py
```

### Menu Options

1. **Browse Model Library**: Explore available models with specifications
2. **Download Model**: Select and download models to local storage
3. **Interactive Chat**: Start conversational sessions with downloaded models
4. **Single Inference**: Perform individual text generation tasks
5. **Manage Models**: View, organize, and delete downloaded models
6. **System Diagnostics**: Check system compatibility and performance
7. **Status Dashboard**: Monitor application status and resource usage
8. **Exit**: Safely terminate the application

### Example Workflows

#### Downloading a Model
1. Select "Browse Model Library" or "Download Model"
2. Choose desired model from the paginated list
3. Monitor download progress with real-time updates
4. Model becomes available for immediate use

#### Starting a Chat Session
1. Select "Interactive Chat"
2. Choose from downloaded models
3. Begin conversation - type messages and receive AI responses
4. Type 'exit' or 'quit' to end session

## Architecture

### Core Components

- **ModelRegistry**: Centralized model metadata and configuration
- **ModelManager**: Handles downloading, storage, and file operations
- **ChatInterface**: Manages conversational interactions and context
- **LlamaCppManager**: Interfaces with llama-cpp-python for inference
- **SystemDiagnostics**: Provides system monitoring and health checks

### Design Principles

- **Modularity**: Clean separation of concerns across components
- **Reliability**: Robust error handling and recovery mechanisms
- **Performance**: Optimized for CPU-only inference with configurable threading
- **User Experience**: Intuitive interface with comprehensive feedback

## Configuration

### Model Parameters

The application uses optimized default parameters:
- **Context Length**: Dynamically adjusted per model
- **Threading**: Limited to 8 threads for stability
- **Batch Size**: 512 tokens for balanced performance
- **Temperature**: 0.8 for natural response generation

### Storage Structure

```
project/
├── models/          # Downloaded model files
├── main.py          # Main application
└── README.md        # Documentation
```

## Technical Specifications

### Dependencies

- **llama-cpp-python**: GGUF model inference
- **urllib**: HTTP operations for model downloads
- **threading**: Concurrent operations
- **pathlib**: Modern file system operations

### Performance Considerations

- Models run on CPU for maximum compatibility
- Memory usage scales with model size and context length
- Download speeds depend on network connectivity
- Inference speed varies by model size and system capabilities

## Troubleshooting

### Common Issues

**Model Loading Failures**
- Verify sufficient system memory
- Check file integrity after download
- Ensure model file permissions are correct

**Download Interruptions**
- Resume functionality handles partial downloads
- Check network connectivity and firewall settings
- Verify sufficient disk space

**Performance Issues**
- Reduce context length for faster inference
- Limit concurrent threads on lower-end systems
- Consider smaller models for resource-constrained environments

## Development

### Project Structure

The codebase follows object-oriented design principles with clear separation of concerns:

- **UI Components**: Color-coded terminal interface with progress indicators
- **Model Management**: Automated downloading and file system operations
- **Inference Engine**: Integration with llama-cpp-python backend
- **System Integration**: Platform-specific optimizations and diagnostics

### Extensibility

The modular architecture supports easy extension:
- Add new models to the ModelRegistry
- Implement additional inference backends
- Extend chat interface with new features
- Add custom system diagnostic checks

## Security and Privacy

- **Offline Operation**: No data transmission after initial model download
- **Local Storage**: All models and conversations stored locally
- **No Telemetry**: No usage data collection or external communications
- **Model Integrity**: Checksum verification for downloaded files

## License

This project is developed for educational and research purposes. Users are responsible for compliance with applicable AI model licenses and terms of use.

## Author

Developed by 0x0806

---

**Version**: 1.1.0  
**Last Updated**: 2025 
**Platform**: Linux/Nix Environment  
**Python Version**: 3.11+
