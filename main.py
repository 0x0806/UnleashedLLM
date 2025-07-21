
#!/usr/bin/env python3
"""
UnleashedLLM - Democratizing Access to Uncensored AI Language Models
Developed by 0x0806

A sophisticated command-line interface tool for downloading, managing, and 
interacting with powerful AI models completely offline.
"""


import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.parse
from datetime import datetime
import hashlib


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class ModelRegistry:
    """Comprehensive registry of uncensored AI models"""
    
    MODELS = {
        # Lightweight Models (1-5)
        "phi-2-uncensored": {
            "name": "Phi-2 Uncensored",
            "size": "1.6GB",
            "category": "Lightweight",
            "description": "Compact uncensored model for basic tasks",
            "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 2048,
            "filename": "phi-2.Q4_K_M.gguf"
        },
        "tinyllama-1b": {
            "name": "TinyLlama 1B Uncensored",
            "size": "0.8GB",
            "category": "Lightweight",
            "description": "Extremely small but capable model",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 2048,
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        },
        "stablelm-zephyr-3b": {
            "name": "StableLM Zephyr 3B Uncensored",
            "size": "2.1GB",
            "category": "Lightweight",
            "description": "Stable and fast small model",
            "url": "https://huggingface.co/TheBloke/zephyr-3B-GGUF/resolve/main/zephyr-3b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 4096,
            "filename": "zephyr-3b.Q4_K_M.gguf"
        },
        "pythia-2.8b": {
            "name": "Pythia 2.8B Uncensored",
            "size": "1.9GB",
            "category": "Lightweight",
            "description": "Efficient model from EleutherAI",
            "url": "https://huggingface.co/TheBloke/pythia-2.8b-GGUF/resolve/main/pythia-2.8b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 2048,
            "filename": "pythia-2.8b.Q4_K_M.gguf"
        },
        "gpt2-xl-uncensored": {
            "name": "GPT-2 XL Uncensored",
            "size": "1.5GB",
            "category": "Lightweight",
            "description": "Classic model without restrictions",
            "url": "https://huggingface.co/TheBloke/GPT2-xl-GGUF/resolve/main/gpt2-xl.Q4_K_M.gguf",
            "capabilities": ["text-generation"],
            "context_length": 1024,
            "filename": "gpt2-xl.Q4_K_M.gguf"
        },

        # Medium Models (6-15)
        "llama2-7b-uncensored": {
            "name": "Llama2 7B Uncensored",
            "size": "4.1GB",
            "category": "Medium",
            "description": "Balanced performance and efficiency",
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "coding"],
            "context_length": 4096,
            "filename": "llama-2-7b-chat.Q4_K_M.gguf"
        },
        "openchat-7b": {
            "name": "OpenChat 7B Uncensored",
            "size": "4.2GB",
            "category": "Medium",
            "description": "High-performance uncensored conversational model",
            "url": "https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF/resolve/main/openchat-3.5-0106.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 8192,
            "filename": "openchat-3.5-0106.Q4_K_M.gguf"
        },
        "mistral-7b-uncensored": {
            "name": "Mistral 7B Uncensored",
            "size": "4.5GB",
            "category": "Medium",
            "description": "High quality 7B parameter model",
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 8192,
            "filename": "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        },
        "wizardlm-7b": {
            "name": "WizardLM 7B Uncensored",
            "size": "4.3GB",
            "category": "Medium",
            "description": "Specialized for complex instructions",
            "url": "https://huggingface.co/TheBloke/WizardLM-7B-V1.0-GGUF/resolve/main/wizardlm-7b-v1.0.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 4096,
            "filename": "wizardlm-7b-v1.0.Q4_K_M.gguf"
        },
        "dolphin-2.6-7b": {
            "name": "Dolphin 2.6 7B Uncensored",
            "size": "4.4GB",
            "category": "Medium",
            "description": "Fine-tuned for uncensored responses",
            "url": "https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF/resolve/main/dolphin-2.6-mistral-7b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 8192,
            "filename": "dolphin-2.6-mistral-7b.Q4_K_M.gguf"
        },
        "neural-chat-7b": {
            "name": "Neural Chat 7B Uncensored",
            "size": "4.3GB",
            "category": "Medium",
            "description": "Optimized for conversational AI",
            "url": "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 4096,
            "filename": "neural-chat-7b-v3-1.Q4_K_M.gguf"
        },
        "openhermes-7b": {
            "name": "OpenHermes 7B Uncensored",
            "size": "4.3GB",
            "category": "Medium",
            "description": "Fine-tuned on diverse datasets",
            "url": "https://huggingface.co/TheBloke/OpenHermes-7B-GGUF/resolve/main/openhermes-7b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 4096,
            "filename": "openhermes-7b.Q4_K_M.gguf"
        },
        "solar-7b": {
            "name": "SOLAR 7B Uncensored",
            "size": "4.5GB",
            "category": "Medium",
            "description": "Efficient 7B model with strong performance",
            "url": "https://huggingface.co/TheBloke/solar-7B-GGUF/resolve/main/solar-7b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "coding"],
            "context_length": 8192,
            "filename": "solar-7b.Q4_K_M.gguf"
        },
        "mistral-7b-openorca": {
            "name": "Mistral 7B OpenOrca Uncensored",
            "size": "4.5GB",
            "category": "Medium",
            "description": "Trained on the OpenOrca dataset",
            "url": "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 8192,
            "filename": "mistral-7b-openorca.Q4_K_M.gguf"
        },
        "zephyr-7b-beta": {
            "name": "Zephyr 7B Beta Uncensored",
            "size": "4.2GB",
            "category": "Medium",
            "description": "Alignment-free version of Zephyr",
            "url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 4096,
            "filename": "zephyr-7b-beta.Q4_K_M.gguf"
        },

        # Large Models (16-25)
        "llama2-13b-uncensored": {
            "name": "Llama2 13B Uncensored",
            "size": "7.8GB",
            "category": "Large",
            "description": "More powerful 13B parameter model",
            "url": "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "coding"],
            "context_length": 4096,
            "filename": "llama-2-13b-chat.Q4_K_M.gguf"
        },
        "codellama-13b": {
            "name": "CodeLlama 13B Uncensored",
            "size": "7.3GB",
            "category": "Large",
            "description": "Specialized for code generation",
            "url": "https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf",
            "capabilities": ["coding", "text-generation"],
            "context_length": 16384,
            "filename": "codellama-13b-instruct.Q4_K_M.gguf"
        },
        "wizardlm-13b": {
            "name": "WizardLM 13B Uncensored",
            "size": "7.9GB",
            "category": "Large",
            "description": "13B version of WizardLM",
            "url": "https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGUF/resolve/main/wizardlm-13b-v1.2.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 4096,
            "filename": "wizardlm-13b-v1.2.Q4_K_M.gguf"
        },
        "nous-hermes-13b": {
            "name": "Nous-Hermes 13B Uncensored",
            "size": "7.8GB",
            "category": "Large",
            "description": "High quality 13B parameter model",
            "url": "https://huggingface.co/TheBloke/Nous-Hermes-13B-GGUF/resolve/main/nous-hermes-13b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 4096,
            "filename": "nous-hermes-13b.Q4_K_M.gguf"
        },
        "chronos-13b": {
            "name": "Chronos 13B Uncensored",
            "size": "7.7GB",
            "category": "Large",
            "description": "Temporal understanding capabilities",
            "url": "https://huggingface.co/TheBloke/chronos-13B-GGUF/resolve/main/chronos-13b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "temporal-reasoning"],
            "context_length": 4096,
            "filename": "chronos-13b.Q4_K_M.gguf"
        },
        "mythomax-13b": {
            "name": "MythoMax 13B Uncensored",
            "size": "7.9GB",
            "category": "Large",
            "description": "Mythological and creative writing",
            "url": "https://huggingface.co/TheBloke/MythoMax-L2-13B-GGUF/resolve/main/mythomax-l2-13b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "creative-writing"],
            "context_length": 4096,
            "filename": "mythomax-l2-13b.Q4_K_M.gguf"
        },
        "xwin-13b": {
            "name": "Xwin 13B Uncensored",
            "size": "7.8GB",
            "category": "Large",
            "description": "Optimized for winning user preferences",
            "url": "https://huggingface.co/TheBloke/Xwin-LM-13B-V0.1-GGUF/resolve/main/xwin-lm-13b-v0.1.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 4096,
            "filename": "xwin-lm-13b-v0.1.Q4_K_M.gguf"
        },
        "airoboros-13b": {
            "name": "Airoboros 13B Uncensored",
            "size": "7.7GB",
            "category": "Large",
            "description": "Fine-tuned on diverse synthetic data",
            "url": "https://huggingface.co/TheBloke/airoboros-13B-gpt4-1.4-GGUF/resolve/main/airoboros-13b-gpt4-1.4.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 4096,
            "filename": "airoboros-13b-gpt4-1.4.Q4_K_M.gguf"
        },
        "orca-13b": {
            "name": "Orca 13B Uncensored",
            "size": "7.8GB",
            "category": "Large",
            "description": "Microsoft's Orca model without restrictions",
            "url": "https://huggingface.co/TheBloke/Orca-2-13B-GGUF/resolve/main/orca-2-13b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 4096,
            "filename": "orca-2-13b.Q4_K_M.gguf"
        },
        "samantha-13b": {
            "name": "Samantha 13B Uncensored",
            "size": "7.7GB",
            "category": "Large",
            "description": "Companion-style conversational AI",
            "url": "https://huggingface.co/TheBloke/Samantha-13B-GGUF/resolve/main/samantha-13b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 4096,
            "filename": "samantha-13b.Q4_K_M.gguf"
        },

        # Extra Large Models (26-30)
        "mixtral-8x7b": {
            "name": "Mixtral 8x7B Uncensored",
            "size": "26.9GB",
            "category": "Extra Large",
            "description": "Mixture of experts model",
            "url": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 32768,
            "filename": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
        },
        "llama2-70b-uncensored": {
            "name": "Llama2 70B Uncensored",
            "size": "39.5GB",
            "category": "Extra Large",
            "description": "Most powerful uncensored Llama2 variant",
            "url": "https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/resolve/main/llama-2-70b-chat.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "coding", "reasoning"],
            "context_length": 4096,
            "filename": "llama-2-70b-chat.Q4_K_M.gguf"
        },
        "goliath-120b": {
            "name": "Goliath 120B Uncensored",
            "size": "61.4GB",
            "category": "Extra Large",
            "description": "Massive uncensored model for research",
            "url": "https://huggingface.co/TheBloke/goliath-120B-GGUF/resolve/main/goliath-120b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "reasoning"],
            "context_length": 8192,
            "filename": "goliath-120b.Q4_K_M.gguf"
        },
        "yi-34b": {
            "name": "Yi 34B Uncensored",
            "size": "20.1GB",
            "category": "Extra Large",
            "description": "High-performance Chinese-English model",
            "url": "https://huggingface.co/TheBloke/Yi-34B-GGUF/resolve/main/yi-34b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat", "multilingual"],
            "context_length": 4096,
            "filename": "yi-34b.Q4_K_M.gguf"
        },
        "dolphin-2.7-mixtral": {
            "name": "Dolphin 2.7 Mixtral Uncensored",
            "size": "27.1GB",
            "category": "Extra Large",
            "description": "Uncensored Mixtral fine-tune",
            "url": "https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF/resolve/main/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
            "capabilities": ["text-generation", "chat"],
            "context_length": 32768,
            "filename": "dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf"
        }
    }

class AnimatedBanner:
    """Animated banner display with ASCII art"""
    
    @staticmethod
    def show_banner():
        banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔                                        ═════════════════════════════════════════════════════════════                        ═╗
║                                                                                                                              ║
║                                          ██╗   ██╗███╗   ██╗██╗     ███████╗ █████╗ ███████╗██╗  ██╗                         ║
║                                          ██║   ██║████╗  ██║██║     ██╔════╝██╔══██╗██╔════╝██║  ██║                         ║
║                                          ██║   ██║██╔██╗ ██║██║     █████╗  ███████║███████╗███████║                         ║
║                                          ██║   ██║██║╚██╗██║██║     ██╔══╝  ██╔══██║╚════██║██╔══██║                         ║
║                                          ╚██████╔╝██║ ╚████║███████╗███████╗██║  ██║███████║██║  ██║                         ║
║                                           ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝                         ║
║                                                                                                                              ║
║                                                                                                      L L M                   ║
║                                                                                                                              ║
║                                                               Democratizing Access to Uncensored AI                          ║
║                                                               Developed by 0x0806                                            ║
╚                                          ══════════════════════════════════════════════════════════════                      ╝
{Colors.END}
        """
        print(banner)
        time.sleep(1)

class ProgressBar:
    """Advanced progress bar with real-time updates"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, amount: int):
        self.current += amount
        self._display()
        
    def _display(self):
        if self.total == 0:
            percentage = 100
        else:
            percentage = min(100, (self.current / self.total) * 100)
        
        bar_length = 50
        filled_length = int(bar_length * percentage // 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if self.current > 0 and elapsed_time > 0:
            speed = self.current / elapsed_time
            eta = (self.total - self.current) / speed if speed > 0 else 0
            eta_str = f"ETA: {int(eta)}s"
            speed_str = f" | {speed / (1024*1024):.1f} MB/s"
        else:
            eta_str = "ETA: --"
            speed_str = ""
        
        print(f'\r{Colors.YELLOW}{self.description}: {Colors.GREEN}|{bar}| {Colors.WHITE}{percentage:.1f}% {eta_str}{speed_str}{Colors.END}', end='', flush=True)

class LlamaCppManager:
    """Manages llama-cpp-python installation and usage"""
    
    def __init__(self):
        self.llama_cpp = None
        self._check_installation()
        
    def _check_installation(self):
        """Check if llama-cpp-python is installed"""
        try:
            import llama_cpp
            self.llama_cpp = llama_cpp
            print(f"{Colors.GREEN}✓ llama-cpp-python found{Colors.END}")
        except ImportError:
            print(f"{Colors.YELLOW}Installing llama-cpp-python...{Colors.END}")
            self._install_llama_cpp()
            
    def _install_llama_cpp(self):
        """Install llama-cpp-python"""
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python", "--no-cache-dir"
            ])
            import llama_cpp
            self.llama_cpp = llama_cpp
            print(f"{Colors.GREEN}✓ llama-cpp-python installed successfully{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to install llama-cpp-python: {e}{Colors.END}")
            print(f"{Colors.YELLOW}Please install manually: pip install llama-cpp-python{Colors.END}")
            
    def load_model(self, model_path: str, **kwargs):
        """Load a model using llama-cpp-python"""
        if not self.llama_cpp:
            raise RuntimeError("llama-cpp-python not available")
            
        try:
            return self.llama_cpp.Llama(
                model_path=model_path,
                verbose=False,
                **kwargs
            )
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to load model: {e}{Colors.END}")
            return None

class ModelManager:
    """Manages model downloads and storage"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def download_model(self, model_id: str) -> bool:
        """Download a model with progress tracking"""
        if model_id not in ModelRegistry.MODELS:
            print(f"{Colors.RED}Error: Model '{model_id}' not found{Colors.END}")
            return False
            
        model = ModelRegistry.MODELS[model_id]
        filename = model['filename']
        filepath = self.models_dir / filename
        
        if filepath.exists():
            print(f"{Colors.YELLOW}Model already exists: {filepath}{Colors.END}")
            return True
            
        print(f"{Colors.CYAN}Downloading {model['name']} ({model['size']})...{Colors.END}")
        print(f"{Colors.BLUE}URL: {model['url']}{Colors.END}")
        
        try:
            # Get file size
            req = urllib.request.Request(model['url'], method='HEAD')
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                
            if total_size == 0:
                print(f"{Colors.YELLOW}Warning: Could not determine file size{Colors.END}")
                total_size = self._get_size_bytes(model['size'])
                
            progress = ProgressBar(total_size, f"Downloading {model['name']}")
            
            # Download file in chunks
            urllib.request.urlretrieve(
                model['url'], 
                filepath, 
                reporthook=lambda block_num, block_size, total: progress.update(block_size) if block_num > 0 else None
            )
                    
            print(f"\n{Colors.GREEN}✓ Download completed: {filepath}{Colors.END}")
            return True
            
        except Exception as e:
            print(f"\n{Colors.RED}✗ Download failed: {e}{Colors.END}")
            if filepath.exists():
                filepath.unlink()
            return False
            
    def _get_size_bytes(self, size_str: str) -> int:
        """Convert size string to bytes"""
        size_str = size_str.upper().replace('B', '')
        if 'G' in size_str:
            return int(float(size_str.replace('G', '')) * 1024 * 1024 * 1024)
        elif 'M' in size_str:
            return int(float(size_str.replace('M', '')) * 1024 * 1024)
        return int(size_str)
        
    def list_downloaded_models(self) -> List[str]:
        """List all downloaded models"""
        downloaded = []
        for model_id, model in ModelRegistry.MODELS.items():
            filepath = self.models_dir / model['filename']
            if filepath.exists():
                downloaded.append(model_id)
        return downloaded
        
    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model"""
        if model_id not in ModelRegistry.MODELS:
            print(f"{Colors.RED}✗ Model not found: {model_id}{Colors.END}")
            return False
            
        filepath = self.models_dir / ModelRegistry.MODELS[model_id]['filename']
        if filepath.exists():
            filepath.unlink()
            print(f"{Colors.GREEN}✓ Deleted: {model_id}{Colors.END}")
            return True
        print(f"{Colors.RED}✗ Model file not found: {model_id}{Colors.END}")
        return False

class ChatInterface:
    """Interactive chat interface with AI models"""
    
    def __init__(self, model_id: str, model_manager: ModelManager):
        self.model_id = model_id
        self.model_manager = model_manager
        self.conversation_history = []
        self.llama_manager = LlamaCppManager()
        self.model = None
        
    def start_chat(self):
        """Start interactive chat session"""
        if self.model_id not in ModelRegistry.MODELS:
            print(f"{Colors.RED}Error: Model not found{Colors.END}")
            return
            
        model_info = ModelRegistry.MODELS[self.model_id]
        model_path = self.model_manager.models_dir / model_info['filename']
        
        if not model_path.exists():
            print(f"{Colors.RED}Error: Model file not found. Please download the model first.{Colors.END}")
            return
            
        print(f"{Colors.CYAN}Loading {model_info['name']}...{Colors.END}")
        
        try:
            self.model = self.llama_manager.load_model(
                str(model_path),
                n_ctx=min(model_info['context_length'], 2048),  # Limit context for better performance
                n_threads=min(os.cpu_count() or 4, 8),  # Limit threads for stability
                n_batch=512,
                n_gpu_layers=0  # CPU-only for compatibility
            )
            
            if not self.model:
                print(f"{Colors.RED}Failed to load model{Colors.END}")
                return
                
            print(f"{Colors.GREEN}✓ Model loaded successfully{Colors.END}")
            print(f"{Colors.CYAN}Starting chat with {model_info['name']}{Colors.END}")
            print(f"{Colors.YELLOW}Type 'exit' to end the conversation{Colors.END}\n")
            
            while True:
                try:
                    user_input = input(f"{Colors.GREEN}You: {Colors.WHITE}")
                    
                    if user_input.lower() in ['exit', 'quit']:
                        break
                        
                    response = self._generate_response(user_input)
                    print(f"{Colors.BLUE}AI: {Colors.WHITE}{response}{Colors.END}\n")
                    
                    self.conversation_history.append({
                        "user": user_input,
                        "ai": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except KeyboardInterrupt:
                    break
                    
        except Exception as e:
            print(f"{Colors.RED}Error loading model: {e}{Colors.END}")
            
        print(f"{Colors.CYAN}Chat session ended{Colors.END}")
        
    def _generate_response(self, prompt: str) -> str:
        """Generate AI response using llama.cpp"""
        if not self.model:
            return "Error: Model not loaded"
            
        try:
            print(f"{Colors.YELLOW}Generating response...{Colors.END}")
            
            # Build conversation context with better formatting
            context = "You are a helpful and knowledgeable AI assistant. Respond naturally and helpfully to user questions.\n\n"
            
            # Add recent conversation history
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges for context
                context += f"Human: {entry['user']}\nAssistant: {entry['ai']}\n\n"
            
            # Add current prompt
            full_prompt = f"{context}Human: {prompt}\nAssistant:"
            
            response = self.model(
                full_prompt,
                max_tokens=256,
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["Human:", "User:", "\nHuman:", "\nUser:"],
                echo=False
            )
            
            result = response['choices'][0]['text'].strip()
            
            # Clean up the response
            if result.startswith("Assistant:"):
                result = result[10:].strip()
            if result.startswith("AI:"):
                result = result[3:].strip()
                
            return result if result else "I understand. How can I help you with that?"
            
        except Exception as e:
            return f"Error generating response: {e}"
            
    def single_inference(self, prompt: str) -> str:
        """Perform single inference without conversation context"""
        if self.model_id not in ModelRegistry.MODELS:
            return "Error: Model not found"
            
        model_info = ModelRegistry.MODELS[self.model_id]
        model_path = self.model_manager.models_dir / model_info['filename']
        
        if not model_path.exists():
            return "Error: Model file not found. Please download the model first."
            
        try:
            if not self.model:
                print(f"{Colors.CYAN}Loading {model_info['name']}...{Colors.END}")
                self.model = self.llama_manager.load_model(
                    str(model_path),
                    n_ctx=model_info['context_length'],
                    n_threads=os.cpu_count() or 4
                )
                
            if not self.model:
                return "Error: Failed to load model"
                
            print(f"{Colors.YELLOW}Processing...{Colors.END}")
            
            # Format prompt better for single inference
            formatted_prompt = f"You are a helpful AI assistant. Please provide a clear and useful response to the following:\n\nHuman: {prompt}\nAssistant:"
            
            response = self.model(
                formatted_prompt,
                max_tokens=400,
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["Human:", "User:", "\nHuman:", "\nUser:"],
                echo=False
            )
            
            result = response['choices'][0]['text'].strip()
            
            # Clean up the response
            if result.startswith("Assistant:"):
                result = result[10:].strip()
            if result.startswith("AI:"):
                result = result[3:].strip()
                
            return result if result else "I understand your request. How can I assist you further?"
            
        except Exception as e:
            return f"Error: {e}"

class SystemDiagnostics:
    """System diagnostic tools for optimal performance"""
    
    @staticmethod
    def check_system_requirements():
        """Check system requirements and performance"""
        print(f"{Colors.CYAN}System Diagnostics{Colors.END}")
        print("=" * 50)
        
        # Check Python version
        python_version = sys.version.split()[0]
        print(f"Python Version: {Colors.GREEN}{python_version}{Colors.END}")
        
        # Check available disk space
        total, used, free = SystemDiagnostics._get_disk_usage()
        print(f"Disk Space - Total: {total}GB, Free: {free}GB, Used: {used}GB")
        
        # Check CPU cores
        cpu_count = os.cpu_count() or 1
        print(f"CPU Cores: {Colors.GREEN}{cpu_count}{Colors.END}")
        
        # Check llama-cpp-python
        try:
            import llama_cpp
            print(f"llama-cpp-python: {Colors.GREEN}Installed{Colors.END}")
        except ImportError:
            print(f"llama-cpp-python: {Colors.RED}Not Installed{Colors.END}")
        
        print("=" * 50)
        
    @staticmethod
    def _get_disk_usage() -> Tuple[int, int, int]:
        """Get disk usage statistics"""
        try:
            statvfs = os.statvfs('.')
            total = (statvfs.f_frsize * statvfs.f_blocks) // (1024**3)
            free = (statvfs.f_frsize * statvfs.f_available) // (1024**3)
            used = total - free
            return total, used, free
        except:
            return 100, 50, 50  # Default values

class PaginatedModelBrowser:
    """Paginated browser for model library"""
    
    def __init__(self, models_per_page: int = 5):
        self.models_per_page = models_per_page
        self.current_page = 0
        
    def display_models(self):
        """Display paginated model library"""
        models = list(ModelRegistry.MODELS.items())
        total_pages = (len(models) - 1) // self.models_per_page + 1
        
        while True:
            self._display_page(models, total_pages)
            
            choice = input(f"\n{Colors.YELLOW}Navigation: [n]ext, [p]revious, [s]elect, [q]uit: {Colors.WHITE}").lower()
            
            if choice == 'n' and self.current_page < total_pages - 1:
                self.current_page += 1
            elif choice == 'p' and self.current_page > 0:
                self.current_page -= 1
            elif choice == 's':
                return self._select_model(models)
            elif choice == 'q':
                return None
                
    def _display_page(self, models: List, total_pages: int):
        """Display current page of models"""
        print(f"\n{Colors.CYAN}Model Library - Page {self.current_page + 1}/{total_pages}{Colors.END}")
        print("=" * 80)
        
        start_idx = self.current_page * self.models_per_page
        end_idx = start_idx + self.models_per_page
        page_models = models[start_idx:end_idx]
        
        for i, (model_id, model) in enumerate(page_models, 1):
            print(f"{Colors.BOLD}{i}. {model['name']}{Colors.END}")
            print(f"   Size: {Colors.GREEN}{model['size']}{Colors.END} | "
                  f"Category: {Colors.YELLOW}{model['category']}{Colors.END}")
            print(f"   {model['description']}")
            print(f"   Capabilities: {', '.join(model['capabilities'])}")
            print(f"   Context Length: {model['context_length']} tokens")
            print()
            
    def _select_model(self, models: List) -> Optional[str]:
        """Select a model from current page"""
        try:
            choice = int(input(f"{Colors.YELLOW}Select model number: {Colors.WHITE}"))
            start_idx = self.current_page * self.models_per_page
            
            if 1 <= choice <= min(self.models_per_page, len(models) - start_idx):
                model_id = models[start_idx + choice - 1][0]
                return model_id
            else:
                print(f"{Colors.RED}Invalid selection{Colors.END}")
                return None
        except ValueError:
            print(f"{Colors.RED}Invalid input{Colors.END}")
            return None

class UnleashedLLM:
    """Main application class"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.browser = PaginatedModelBrowser()
        
    def run(self):
        """Main application loop"""
        AnimatedBanner.show_banner()
        
        while True:
            self._display_menu()
            choice = input(f"{Colors.YELLOW}Select option: {Colors.WHITE}").strip()
            
            if choice == '1':
                self._browse_models()
            elif choice == '2':
                self._download_model()
            elif choice == '3':
                self._chat_with_model()
            elif choice == '4':
                self._single_inference()
            elif choice == '5':
                self._manage_models()
            elif choice == '6':
                SystemDiagnostics.check_system_requirements()
            elif choice == '7':
                self._show_status_dashboard()
            elif choice == '8':
                print(f"{Colors.CYAN}Thank you for using UnleashedLLM!{Colors.END}")
                break
            else:
                print(f"{Colors.RED}Invalid option. Please try again.{Colors.END}")
                
            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
            
    def _display_menu(self):
        """Display main menu"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}UnleashedLLM Main Menu{Colors.END}")
        print("=" * 40)
        print(f"{Colors.WHITE}1. Browse Model Library")
        print("2. Download Model")
        print("3. Interactive Chat")
        print("4. Single Inference")
        print("5. Manage Models")
        print("6. System Diagnostics")
        print("7. Status Dashboard")
        print(f"8. Exit{Colors.END}")
        print("=" * 40)
        
    def _browse_models(self):
        """Browse available models"""
        print(f"\n{Colors.CYAN}Browsing Model Library{Colors.END}")
        self.browser.display_models()
        
    def _download_model(self):
        """Download a selected model"""
        print(f"\n{Colors.CYAN}Download Model{Colors.END}")
        model_id = self.browser.display_models()
        
        if model_id:
            self.model_manager.download_model(model_id)
            
    def _chat_with_model(self):
        """Start interactive chat"""
        downloaded = self.model_manager.list_downloaded_models()
        
        if not downloaded:
            print(f"{Colors.RED}No models downloaded. Please download a model first.{Colors.END}")
            return
            
        print(f"\n{Colors.CYAN}Available Models:{Colors.END}")
        for i, model_id in enumerate(downloaded, 1):
            model = ModelRegistry.MODELS.get(model_id, {})
            print(f"{i}. {model.get('name', model_id)}")
            
        try:
            choice = int(input(f"{Colors.YELLOW}Select model: {Colors.WHITE}")) - 1
            if 0 <= choice < len(downloaded):
                chat = ChatInterface(downloaded[choice], self.model_manager)
                chat.start_chat()
            else:
                print(f"{Colors.RED}Invalid selection{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Invalid input{Colors.END}")
            
    def _single_inference(self):
        """Perform single inference"""
        downloaded = self.model_manager.list_downloaded_models()
        
        if not downloaded:
            print(f"{Colors.RED}No models downloaded. Please download a model first.{Colors.END}")
            return
            
        print(f"\n{Colors.CYAN}Available Models:{Colors.END}")
        for i, model_id in enumerate(downloaded, 1):
            model = ModelRegistry.MODELS.get(model_id, {})
            print(f"{i}. {model.get('name', model_id)}")
            
        try:
            choice = int(input(f"{Colors.YELLOW}Select model: {Colors.WHITE}")) - 1
            if 0 <= choice < len(downloaded):
                prompt = input(f"{Colors.YELLOW}Enter your prompt: {Colors.WHITE}")
                
                chat = ChatInterface(downloaded[choice], self.model_manager)
                response = chat.single_inference(prompt)
                print(f"{Colors.GREEN}Response: {response}{Colors.END}")
            else:
                print(f"{Colors.RED}Invalid selection{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Invalid input{Colors.END}")
        
    def _manage_models(self):
        """Manage downloaded models"""
        downloaded = self.model_manager.list_downloaded_models()
        
        if not downloaded:
            print(f"{Colors.RED}No models downloaded{Colors.END}")
            return
            
        print(f"\n{Colors.CYAN}Model Management{Colors.END}")
        print("Downloaded Models:")
        
        total_size = 0
        for i, model_id in enumerate(downloaded, 1):
            model = ModelRegistry.MODELS.get(model_id, {})
            filepath = self.model_manager.models_dir / model['filename']
            if filepath.exists():
                file_size = filepath.stat().st_size / (1024**3)  # GB
                total_size += file_size
                print(f"{i}. {model.get('name', model_id)} ({file_size:.1f}GB)")
            
        print(f"\nTotal storage used: {total_size:.1f}GB")
            
        choice = input(f"\n{Colors.YELLOW}[d]elete model, [l]ist all, [q]uit: {Colors.WHITE}").lower()
        
        if choice == 'd':
            try:
                idx = int(input(f"{Colors.YELLOW}Select model to delete: {Colors.WHITE}")) - 1
                if 0 <= idx < len(downloaded):
                    self.model_manager.delete_model(downloaded[idx])
                else:
                    print(f"{Colors.RED}Invalid selection{Colors.END}")
            except ValueError:
                print(f"{Colors.RED}Invalid input{Colors.END}")
                
    def _show_status_dashboard(self):
        """Show real-time status dashboard"""
        print(f"\n{Colors.CYAN}Status Dashboard{Colors.END}")
        print("=" * 50)
        
        downloaded = self.model_manager.list_downloaded_models()
        print(f"Downloaded Models: {Colors.GREEN}{len(downloaded)}{Colors.END}")
        print(f"Available Models: {Colors.BLUE}{len(ModelRegistry.MODELS)}{Colors.END}")
        
        total_size = 0
        for model_id in downloaded:
            model = ModelRegistry.MODELS.get(model_id, {})
            filepath = self.model_manager.models_dir / model['filename']
            if filepath.exists():
                total_size += filepath.stat().st_size / (1024**3)  # GB
                
        print(f"Storage Used: {Colors.YELLOW}{total_size:.1f}GB{Colors.END}")
        
        # Check llama.cpp status
        try:
            import llama_cpp
            print(f"llama-cpp-python: {Colors.GREEN}Ready{Colors.END}")
        except ImportError:
            print(f"llama-cpp-python: {Colors.RED}Not Installed{Colors.END}")
            
        print("=" * 50)

def main():
    """Entry point"""
    try:
        app = UnleashedLLM()
        app.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}Application terminated by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.END}")

if __name__ == "__main__":
    main()
