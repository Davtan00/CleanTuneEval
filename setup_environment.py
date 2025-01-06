import subprocess
import sys
import os
from pathlib import Path
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
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
        print(f"Is MPS available? {torch.backends.mps.is_available()}")
        print(f"Is CUDA available? {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not yet installed")

def install_base_torch(is_apple_silicon):
    """Install the base PyTorch version needed for dependency resolution."""
    print("\nInstalling base PyTorch...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch==2.5.1", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing PyTorch: {e}")
        return False

def setup_requirements():
    """Set up requirements files for different platforms."""
    print("\n=== Setting up requirements ===")
    
    # First, ensure we have pip-tools without assuming it exists
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        subprocess.run([
            sys.executable, "-m", "pip", "install", "pip-tools"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error installing pip-tools: {e}")
        return False
    
    # Generate base requirements.txt without hardware-specific packages
    print("\nGenerating base requirements.txt...")
    try:
        subprocess.run([
            "pip-compile",
            "requirements.in",
            "--upgrade",
            "--strip-extras",
            # Exclude hardware-specific packages from base requirements
            "--constraint", "constraints.txt",
            "--output-file", "requirements.txt"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating requirements.txt: {e}")
        return False

    # Create platform-specific requirements
    requirements_dir = Path('requirements')
    requirements_dir.mkdir(exist_ok=True)
    
    platform_specs = {
        'cpu': """# CPU-specific requirements
-r ../requirements.txt
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
--index-url https://download.pytorch.org/whl/cpu
""",
        'cuda': """# CUDA-specific requirements
-r ../requirements.txt
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
bitsandbytes>=0.42.0  # Only for CUDA
--index-url https://download.pytorch.org/whl/cu118
""",
        'mps': """# MPS-specific requirements (Apple Silicon)
-r ../requirements.txt
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
# Note: bitsandbytes is installed but won't have GPU support
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
    
    # Install base requirements first (this includes most dependencies)
    print("\nInstalling base requirements...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    # Now handle PyTorch specifically for the platform
    print("\nInstalling platform-specific PyTorch...")
    if is_apple_silicon:
        print("Installing PyTorch for Apple Silicon...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch==2.5.1", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"  # CPU build works with MPS
        ])
    else:
        # Import torch here since it should be installed by now
        import torch
        if torch.cuda.is_available():
            print("Installing CUDA-enabled PyTorch...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch==2.5.1", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
        else:
            print("Installing CPU-only PyTorch...")
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

def verify_mps_compatibility():
    """Verify MPS compatibility for critical operations."""
    try:
        import torch
        if not torch.backends.mps.is_available():
            return
            
        device = torch.device('mps')
        
        # Test basic tensor operations
        a = torch.ones(2, 2).to(device)
        b = torch.ones(2, 2).to(device)
        _ = torch.matmul(a, b)
        print("✓ Basic tensor operations work on MPS")
        
        # Test neural network operations
        x = torch.randn(1, 3, 224, 224).to(device)
        conv = torch.nn.Conv2d(3, 64, 3).to(device)
        _ = conv(x)
        print("✓ Neural network operations work on MPS")
        
    except Exception as e:
        print(f"⚠️  MPS compatibility issue detected: {e}")

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
    
    # Install base PyTorch first (needed for dependency resolution)
    if not install_base_torch(is_apple_silicon):
        print("Failed to install base PyTorch. Exiting.")
        return
    
    if not setup_requirements():
        print("Failed to setup requirements. Exiting.")
        return
        
    install_dependencies(is_apple_silicon)
    verify_installation()
    
    if is_apple_silicon:
        verify_mps_compatibility()
        # Only import HardwareConfig after all dependencies are installed
        try:
            from src.config.environment import HardwareConfig
            hardware_config = HardwareConfig()
            hardware_config.check_mps_limitations()
        except ImportError as e:
            print(f"Note: Could not import HardwareConfig: {e}")
            print("This is expected if src package is not in PYTHONPATH")
    
    print("\nSetup complete! Please restart your Python environment.")

if __name__ == "__main__":
    main() 