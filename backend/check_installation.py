#!/usr/bin/env python3
"""
Installation Verification Script
=================================

Run this to check if all Day 1 dependencies are installed correctly.

Usage:
    python check_installation.py
"""

import sys

print("="*70)
print("NOONCHI TRANSLATOR - INSTALLATION CHECK")
print("="*70)
print()

all_good = True

# Check Python version
print("1. Checking Python version...")
version = sys.version_info
if version.major >= 3 and version.minor >= 9:
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"   ✗ Python {version.major}.{version.minor}.{version.micro}")
    print(f"   ⚠️  Need Python 3.9+")
    all_good = False
print()

# Check spaCy
print("2. Checking spaCy...")
try:
    import spacy
    print(f"   ✓ spaCy {spacy.__version__} installed")
    
    # Check for English model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("   ✓ English model (en_core_web_sm) loaded")
    except OSError:
        print("   ✗ English model not found")
        print("   ⚠️  Run: python -m spacy download en_core_web_sm")
        all_good = False
except ImportError:
    print("   ✗ spaCy not installed")
    print("   ⚠️  Run: pip install spacy")
    all_good = False
print()

# Check KoNLPy
print("3. Checking KoNLPy...")
try:
    import konlpy
    print(f"   ✓ KoNLPy installed")
    
    # Check Java
    try:
        from konlpy.tag import Okt
        okt = Okt()
        print("   ✓ Java found (KoNLPy works)")
    except Exception as e:
        print(f"   ✗ Java issue: {e}")
        print("   ⚠️  Install Java JDK")
        all_good = False
        
except ImportError:
    print("   ✗ KoNLPy not installed")
    print("   ⚠️  Run: pip install konlpy")
    all_good = False
print()

# Check Mecab (optional but recommended)
print("4. Checking Mecab (optional)...")
try:
    from konlpy.tag import Mecab
    mecab = Mecab()
    test = mecab.morphs("테스트")
    print("   ✓ Mecab installed and working")
    print("   🎉 This is the fastest tokenizer!")
except ImportError:
    print("   ⚠️  Mecab not installed (this is OK for Day 1)")
    print("   ℹ️  Can use Okt instead (slower but works)")
    print("   ℹ️  To install Mecab:")
    print("      Mac: brew install mecab mecab-ko mecab-ko-dic")
    print("      Ubuntu: apt-get install mecab libmecab-dev")
except Exception as e:
    print(f"   ⚠️  Mecab error: {e}")
    print("   ℹ️  Can use Okt instead for Day 1")
print()

# Check korean-conjugator
print("5. Checking korean-conjugator...")
try:
    import korean_conjugator
    print("   ✓ korean-conjugator installed")
except ImportError:
    print("   ✗ korean-conjugator not installed")
    print("   ⚠️  Run: pip install korean-conjugator")
    all_good = False
print()

# Check FastAPI (for later, but good to verify)
print("6. Checking FastAPI (for Days 10-11)...")
try:
    import fastapi
    print(f"   ✓ FastAPI {fastapi.__version__} installed")
except ImportError:
    print("   ⚠️  FastAPI not installed (needed later)")
    print("   ℹ️  Run: pip install fastapi uvicorn")
print()

# Summary
print("="*70)
if all_good:
    print("🎉 ALL CRITICAL DEPENDENCIES INSTALLED!")
    print()
    print("You're ready to run Day 1 lab:")
    print("  cd backend/nlp")
    print("  python day1_tokenization_lab.py")
else:
    print("⚠️  SOME DEPENDENCIES MISSING")
    print()
    print("Please install missing dependencies above.")
    print("See docs/DAY_1_SETUP.md for detailed instructions.")
    
print("="*70)
print()

# Bonus: Quick tokenization test
if all_good:
    print("BONUS: Quick Tokenization Test")
    print("-"*70)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("I want to eat")
        print(f"English: 'I want to eat'")
        print(f"Tokens: {[token.text for token in doc]}")
        print()
        
        try:
            from konlpy.tag import Mecab
            mecab = Mecab()
            print(f"Korean: '먹고 싶어요'")
            print(f"Morphemes: {mecab.morphs('먹고 싶어요')}")
        except:
            from konlpy.tag import Okt
            okt = Okt()
            print(f"Korean: '먹고 싶어요'")
            print(f"Morphemes: {okt.morphs('먹고 싶어요')}")
        
        print()
        print("✓ Tokenization working! Ready for NLP! 🚀")
        print()
    except Exception as e:
        print(f"Test failed: {e}")
        print()
