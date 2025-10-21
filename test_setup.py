"""
Quick test script to verify SmartClaim setup.
Run this to check if all dependencies are installed correctly.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("shap", "shap"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("streamlit", "streamlit"),
        ("joblib", "joblib"),
    ]
    
    failed = []
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - NOT FOUND")
            failed.append(package_name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages installed correctly!")
        return True


def test_project_structure():
    """Test that project directories exist."""
    print("\nTesting project structure...")
    
    required_dirs = [
        "src",
        "src/data",
        "src/eda",
        "src/models",
        "src/app",
        "data",
        "data/raw",
        "data/processed",
        "artifacts",
        "artifacts/models",
        "artifacts/encoders",
        "artifacts/reports",
        "artifacts/reports/plots",
    ]
    
    failed = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - NOT FOUND")
            failed.append(dir_path)
    
    if failed:
        print(f"\n❌ Missing directories: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All directories exist!")
        return True


def test_module_imports():
    """Test that custom modules can be imported."""
    print("\nTesting custom module imports...")
    
    modules = [
        "src.data.generate_synthetic",
        "src.data.load_data",
        "src.models.utils",
        "src.models.train",
        "src.models.evaluate",
        "src.models.explain",
    ]
    
    failed = []
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            print(f"  ✗ {module_name} - ERROR: {e}")
            failed.append(module_name)
    
    if failed:
        print(f"\n⚠ Some modules failed to import")
        print("   This might be OK if you haven't generated data yet")
        return True  # Don't fail on this
    else:
        print("\n✓ All modules import successfully!")
        return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("Setup Test Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Generate data:")
    print("     python -m src.data.generate_synthetic")
    print("\n  2. Train models:")
    print("     python -m src.models.train")
    print("\n  3. Evaluate models:")
    print("     python -m src.models.evaluate")
    print("\n  4. Generate explanations:")
    print("     python -m src.models.explain")
    print("\n  5. Run the app:")
    print("     streamlit run src/app/streamlit_app.py")
    print("\nFor more details, see README.md")
    print("="*60)


def main():
    """Run all tests."""
    print("="*60)
    print("SmartClaim Setup Test")
    print("="*60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test project structure
    if not test_project_structure():
        all_passed = False
    
    # Test module imports
    if not test_module_imports():
        all_passed = False
    
    # Print next steps
    if all_passed:
        print_next_steps()
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

