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
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--upgrade", "pip", "pip-tools"
    ])
    
    # Generate base requirements.txt
    print("\nGenerating base requirements.txt...")
    try:
        subprocess.run([
            "pip-compile",
            "requirements.in",
            "--upgrade",
            "--output-file", "requirements.txt"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating requirements.txt: {e}")
        return False

    # Create platform-specific requirements
    requirements_dir = Path('requirements')
    requirements_dir.mkdir(exist_ok=True)
    
    # Create platform-specific requirements manually
    platform_specs = {
        'cpu': """# CPU-specific requirements
-r ../requirements.txt
torch==2.5.1
torchvision
torchaudio
--index-url https://download.pytorch.org/whl/cpu
""",
        'cuda': """# CUDA-specific requirements
-r ../requirements.txt
torch==2.5.1
torchvision
torchaudio
--index-url https://download.pytorch.org/whl/cu118
""",
        'mps': """# MPS-specific requirements (Apple Silicon)
-r ../requirements.txt
torch==2.5.1
torchvision
torchaudio
--index-url https://download.pytorch.org/whl/cpu
"""
    }
    
    for platform_type, content in platform_specs.items():
        req_file = requirements_dir / f"requirements-{platform_type}.txt"
        print(f"\nGenerating {req_file}...")
        try:
            with open(req_file, 'w') as f:
                f.write(content)
            print(f"Created {req_file}")
        except Exception as e:
            print(f"Error creating {req_file}: {e}")
            return False
    
    return True

def install_dependencies(is_apple_silicon):
    """Install dependencies based on platform."""
    print("\n=== Installing Dependencies ===")
    
    # First uninstall potentially conflicting packages
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall", "-y",
        "torch", "torchvision", "torchaudio"
    ])
    
    # Install base requirements first
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    # Install platform-specific PyTorch
    if is_apple_silicon:
        print("\nInstalling PyTorch for Apple Silicon...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch==2.5.1", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"  # CPU build works with MPS
        ])
    else:
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