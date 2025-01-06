import subprocess
import sys
import os
from pathlib import Path
import torch
import platform

def check_system():
    """Check system configuration and requirements."""
    print("\n=== System Check ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if running on Apple Silicon
    is_apple_silicon = platform.processor() == 'arm'
    print(f"Running on Apple Silicon: {is_apple_silicon}")
    
    return is_apple_silicon

def check_torch_installation():
    """Verify PyTorch installation and available devices."""
    print("\n=== PyTorch Configuration ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")

def setup_requirements():
    """Set up requirements files for different platforms."""
    print("\n=== Setting up requirements ===")
    
    # Ensure pip and pip-tools are up to date
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "pip-tools"])
    
    # Generate platform-specific requirements
    requirements = {
        'base': 'requirements.txt',
        'cpu': 'requirements/requirements-cpu.txt',
        'cuda': 'requirements/requirements-cuda.txt',
        'mps': 'requirements/requirements-mps.txt'
    }
    
    # Create requirements directory if it doesn't exist
    Path('requirements').mkdir(exist_ok=True)
    
    for platform_type, req_file in requirements.items():
        print(f"\nGenerating {req_file}...")
        try:
            if platform_type == 'base':
                subprocess.run([
                    "pip-compile",
                    "requirements.in",
                    "--upgrade",
                    "--output-file", req_file
                ], check=True)
            else:
                subprocess.run([
                    "pip-compile",
                    "requirements.in",
                    "--upgrade",
                    "--output-file", req_file,
                    f"--extra={platform_type}"
                ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating {req_file}: {e}")

def install_dependencies(is_apple_silicon):
    """Install dependencies based on platform."""
    print("\n=== Installing Dependencies ===")
    
    # First uninstall potentially conflicting packages
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall", "-y",
        "torch", "torchvision", "torchaudio"
    ])
    
    if is_apple_silicon:
        print("\nInstalling PyTorch for Apple Silicon...")
        # Install PyTorch with MPS support
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch==2.5.1", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
    else:
        # For non-Apple Silicon, use appropriate requirements
        if torch.cuda.is_available():
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch==2.5.1", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
        else:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch==2.5.1", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ])

def verify_installation():
    """Verify the installation was successful."""
    print("\n=== Verifying Installation ===")
    
    # Check key packages
    packages = [
        "torch",
        "sentence_transformers",
        "transformers",
        "datasets",
        "accelerate"
    ]
    
    for package in packages:
        try:
            module = __import__(package)
            print(f"✓ {package} installed successfully")
        except ImportError as e:
            print(f"✗ Error importing {package}: {e}")

def main():
    """Main setup function."""
    print("Starting environment setup...")
    
    is_apple_silicon = check_system()
    check_torch_installation()
    
    # Ask for confirmation before proceeding
    proceed = input("\nProceed with setup? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Setup cancelled.")
        return
    
    setup_requirements()
    install_dependencies(is_apple_silicon)
    verify_installation()
    
    print("\nSetup complete! Please restart your Python environment.")

if __name__ == "__main__":
    main() 