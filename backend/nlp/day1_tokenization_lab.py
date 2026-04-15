"""
DAY 1 LAB: Understanding Tokenization
======================================

Welcome to your first NLP hands-on lab!

NLP CONCEPT: Tokenization
--------------------------
Tokenization = Breaking text into meaningful units (tokens)

Why it matters: 
You can't process language without breaking it down first!
It's like trying to cook without chopping ingredients.

LEARNING GOALS:
1. Understand what tokenization is
2. See how English and Korean tokenization differ
3. Learn why Korean needs morphological analysis
4. Prepare for building translation systems

INSTRUCTIONS:
1. Make sure you've installed: spacy, konlpy
2. Run this file: python day1_tokenization_lab.py
3. Observe the outputs carefully
4. Answer the checkpoint questions at the end
"""

# Part 1: English Tokenization with spaCy
print("="*70)
print("PART 1: ENGLISH TOKENIZATION")
print("="*70)

try:
    import spacy
    
    # Load English language model
    nlp = spacy.load("en_core_web_sm")
    
    def explore_english_tokenization():
        """See how English tokenization works"""
        
        sentence = "I want to eat delicious food"
        doc = nlp(sentence)
        
        print(f"\nOriginal sentence: '{sentence}'\n")
        
        # Simple tokens (words)
        print("TOKENS (words):")
        tokens = [token.text for token in doc]
        print(f"  {tokens}")
        print(f"  Total: {len(tokens)} tokens\n")
        
        # Lemmas (dictionary/base forms)
        print("LEMMAS (dictionary forms):")
        lemmas = [token.lemma_ for token in doc]
        print(f"  {lemmas}")
        print("  Note: 'want' stays 'want', but 'eating' → 'eat'\n")
        
        # POS tags (grammatical roles)
        print("POS TAGS (what role each word plays):")
        for token in doc:
            print(f"  {token.text:12} → {token.pos_:8} ({token.tag_:5}) - {spacy.explain(token.pos_)}")
        
        print("\n  Key POS tags:")
        print("    PRON = Pronoun (I, you, he)")
        print("    VERB = Verb (eat, go, want)")
        print("    PART = Particle (to)")
        print("    ADJ  = Adjective (delicious)")
        print("    NOUN = Noun (food)")
        
        # Dependencies (sentence structure)
        print("\nDEPENDENCIES (how words relate):")
        for token in doc:
            print(f"  {token.text:12} --{token.dep_:10}--> {token.head.text}")
        
        print("\n  This shows the sentence structure!")
        print("  'want' is the ROOT (main verb)")
        print("  'I' is the subject (nsubj) of 'want'")
        print("  'eat' is what you want (xcomp)")
        print("  'food' is what you eat (dobj)")
    
    explore_english_tokenization()

except ImportError:
    print("\n⚠️  spaCy not installed!")
    print("Install with: pip install spacy")
    print("Then download model: python -m spacy download en_core_web_sm")


# Part 2: Korean Tokenization with KoNLPy
print("\n\n" + "="*70)
print("PART 2: KOREAN TOKENIZATION - THE GAME CHANGER")
print("="*70)

try:
    from konlpy.tag import Mecab
    
    def explore_korean_tokenization():
        """See how Korean tokenization is DIFFERENT"""
        
        mecab = Mecab()
        
        # Same meaning, three formality levels
        print("\nSAME MEANING, DIFFERENT FORMALITY:")
        print("(All mean: 'want to eat')\n")
        
        sentences = {
            'CASUAL (friends)': '먹고 싶어',
            'POLITE (colleagues)': '먹고 싶어요',
            'FORMAL (boss)': '먹고 싶습니다'
        }
        
        for formality, sentence in sentences.items():
            print(f"{formality}")
            print(f"  Korean: {sentence}")
            
            # Morphemes (smallest meaning units)
            morphs = mecab.morphs(sentence)
            print(f"  Morphemes: {morphs}")
            
            # POS tags
            pos = mecab.pos(sentence)
            print(f"  POS tags: {pos}")
            
            # Explain what each morpheme means
            print(f"  Breakdown:")
            if '먹' in morphs:
                print(f"    먹 = 'eat' (verb stem)")
            if '고' in morphs:
                print(f"    고 = connecting ending")
            if '싶' in morphs:
                print(f"    싶 = 'want' (desire)")
            if '어' in morphs or '어요' in sentence:
                if '어' in morphs:
                    print(f"    어 = casual ending")
                elif '어요' in sentence:
                    print(f"    어요 = polite ending")
            if '습니다' in sentence:
                print(f"    습니다 = formal ending")
            print()
    
    explore_korean_tokenization()

except ImportError:
    print("\n⚠️  KoNLPy not installed!")
    print("Install with: pip install konlpy")
    print("Note: KoNLPy requires Java JDK to be installed")
    print("\nFor Mecab:")
    print("  Mac: brew install mecab mecab-ko mecab-ko-dic")
    print("  Ubuntu: apt-get install mecab libmecab-dev mecab-ipadic-utf8")
except Exception as e:
    print(f"\n⚠️  Error running Mecab: {e}")
    print("Mecab might not be installed on your system.")
    print("You can try using Okt instead (slower but no system dependencies):")
    print("  from konlpy.tag import Okt")


# Part 3: Key Differences
print("\n" + "="*70)
print("PART 3: WHY KOREAN IS HARDER FOR NLP")
print("="*70)

print("""
ENGLISH (Analytic Language):
  Sentence: "I wanted to eat"
  → Words stay SEPARATE
  → Tense shown with auxiliary verbs ('wanted')
  → Easy to split on spaces
  
  For translation: Can mostly translate word-by-word
  Example: "I" → "나는", "eat" → "먹다", "food" → "음식"

KOREAN (Agglutinative Language):
  Sentence: "먹고 싶었어요"
  → Morphemes STICK TOGETHER
  → 먹 (eat) + 고 싶 (want to) + 었 (past) + 어요 (polite)
  → Can't just split on spaces!
  
  For translation: Must analyze morphemes and reassemble
  Example: Must split "먹었어요" into parts, transform, rebuild

THIS IS WHY WE NEED MORPHOLOGICAL ANALYSIS!

WHAT THIS MEANS FOR YOUR TRANSLATOR:
┌──────────────────────────────────────────────────────┐
│ To change formality in Korean:                       │
│                                                       │
│ 1. SPLIT: 먹었어요 → [먹, 었, 어요]                   │
│ 2. TRANSFORM: 어요 → 습니다 (formal)                  │
│ 3. REASSEMBLE: 먹었습니다                             │
│                                                       │
│ You can't just swap words like in English!           │
└──────────────────────────────────────────────────────┘
""")


# Part 4: Hands-On Challenge
print("="*70)
print("PART 4: YOUR TURN - HANDS-ON CHALLENGE")
print("="*70)

print("""
Try to answer these questions based on what you observed:

CHECKPOINT QUESTIONS:
━━━━━━━━━━━━━━━━━━━━

1. What's the difference between a TOKEN and a MORPHEME?
   Hint: Look at how "eating" was tokenized vs "먹고 싶어요"

2. In Korean, what part of the word changes with formality?
   Compare: 먹고 싶어 vs 먹고 싶어요 vs 먹고 싶습니다

3. Why can't you translate Korean word-by-word like English?

4. How would you extract just the verb stem from "먹고 싶어요"?
   (Think about what the tokenizer showed you)

5. If you wanted to change "먹고 싶어요" (polite) to formal,
   which part would you replace?

EXTRA CHALLENGE:
Try running the Korean tokenizer on these sentences:
  - "가고 싶어요" (want to go)
  - "먹었어요" (ate)
  - "공부했습니다" (studied - formal)
  
What patterns do you notice?
""")


# Summary
print("\n" + "="*70)
print("🎓 DAY 1 COMPLETE!")
print("="*70)

print("""
WHAT YOU LEARNED TODAY:
━━━━━━━━━━━━━━━━━━━━━━

✓ What tokenization is and why it matters
✓ How to use spaCy for English NLP
✓ How to use KoNLPy for Korean NLP
✓ Key difference: English (analytic) vs Korean (agglutinative)
✓ Why morphological analysis is crucial for Korean
✓ Foundation for building your translator

NEXT STEPS:
━━━━━━━━━━

Tomorrow (Day 2), you'll build an English analyzer that:
  • Parses English sentences
  • Extracts subjects, verbs, objects
  • Prepares for translation

This is the FIRST step in your translation pipeline!

KEY INSIGHT TO REMEMBER:
  "You can't translate what you can't parse.
   You can't parse what you can't tokenize."
   
   Tokenization is the foundation of ALL NLP!

HOMEWORK (Optional):
  1. Install all the dependencies if you haven't
  2. Run this file multiple times with different sentences
  3. Try to predict what tokens you'll get before running
  4. Read about agglutinative languages (Korean, Turkish, Japanese)
""")

print("\n" + "="*70)
print("Ready for Day 2? 🚀")
print("="*70 + "\n")
from konlpy.tag import Mecab
mecab = Mecab()
texts = "안녕하세요. 저는 한국어를 공부합니다."
print(f"tokens: {mecab.pos(texts)}")