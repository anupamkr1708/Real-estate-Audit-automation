#!/usr/bin/env python3
"""
ONE-CLICK SETUP SCRIPT
Verifies all requirements and prepares the system

Usage:
    python setup.py
"""

import sys
import subprocess
import os
from pathlib import Path


def print_header(text):
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)


def check_python_version():
    """Check Python version"""
    print_header(" Checking Python Version")
    
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("    ERROR: Python 3.7 or higher required")
        return False
    
    print("    Python version OK")
    return True


def check_required_packages():
    """Check and install required packages"""
    print_header(" Checking Required Packages")
    
    required = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'openpyxl': 'openpyxl',
    'pdfplumber': 'pdfplumber',
    'xlrd': 'xlrd==2.0.2'
}

    
    missing = []
    
    for package, pip_name in required.items():
        try:
            __import__(package)
            print(f"    {package}")
        except ImportError:
            print(f"    {package} - Not installed")
            missing.append(pip_name)
    
    if missing:
        print(f"\n     Missing packages: {', '.join(missing)}")
        response = input("\n   Install missing packages? [Y/n]: ").strip().lower()
        
        if response in ['', 'y', 'yes']:
            print("\n   Installing packages...")
            for package in missing:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("    All packages installed")
        else:
            print("     Skipping installation")
            return False
    else:
        print("\n    All packages installed")
    
    return True


def check_required_files():
    """Check if all required Python files exist"""
    print_header(" Checking Required Files")
    
    required_files = [
        'master_orchestrator.py',
        'auto_discovery_agent.py',
        'combined_extractor.py',
        'pdf_extractor.py',
        'validator.py',
        'calculator.py',
        'reporter.py'
    ]
    
    missing = []
    
    for filename in required_files:
        if Path(filename).exists():
            print(f"    {filename}")
        else:
            print(f"    {filename} - NOT FOUND")
            missing.append(filename)
    
    if missing:
        print(f"\n    ERROR: Missing files: {', '.join(missing)}")
        print("   Please ensure all required files are in the same directory")
        return False
    
    print("\n    All required files present")
    return True


def create_directory_structure():
    """Create necessary directories"""
    print_header(" Creating Directory Structure")
    
    directories = [
        'data',
        'results',
        'results/extracted_xls',
        'results/extracted_pdf',
        'results/validated',
        'results/findings',
        'results/reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"    {directory}/")
    
    print("\n    Directory structure created")
    return True


def check_data_files():
    """Check if data directory has files"""
    print_header(" Checking Data Directory")
    
    data_dir = Path('data')
    
    xls_files = list(data_dir.glob("*.xls")) + list(data_dir.glob("*.xlsx"))
    pdf_files = list(data_dir.glob("*.pdf"))
    
    print(f"\n   Found {len(xls_files)} Excel file(s)")
    print(f"   Found {len(pdf_files)} PDF file(s)")
    
    if not xls_files and not pdf_files:
        print("\n     WARNING: No documents found in data/ directory")
        print("   Please add your lease documents to the data/ directory")
        return False
    
    print("\n    Documents found")
    return True


def print_next_steps():
    """Print next steps"""
    print_header(" Setup Complete!")
    
    print("""
    All requirements verified
    Directory structure created
   
   NEXT STEPS:
   
   1. Put your lease documents in the data/ directory:
      • Prior AE Excel files
      • Owner AE Excel files
      • Rent Roll PDFs
      • Stacking Plan PDFs
   
   2. Run the pipeline:
      
      python master_orchestrator.py
   
   3. Check results in results/reports/ directory
   
   For more information, see the documentation!
   """)


def main():
    """Main setup routine"""
    
    print("\n" + "="*80)
    print(" "*20 + " AGENTIC AUDIT AUTOMATION SETUP")
    print("="*80)
    
    all_ok = True
    
    # Check Python version
    if not check_python_version():
        all_ok = False
    
    # Check packages
    if not check_required_packages():
        all_ok = False
    
    # Check required files
    if not check_required_files():
        all_ok = False
    
    # Create directories
    if not create_directory_structure():
        all_ok = False
    
    # Check data files (warning only)
    check_data_files()
    
    # Print summary
    if all_ok:
        print_next_steps()
        sys.exit(0)
    else:
        print_header(" Setup Failed")
        print("\n   Please fix the errors above and run setup again")
        sys.exit(1)


if __name__ == "__main__":
    main()