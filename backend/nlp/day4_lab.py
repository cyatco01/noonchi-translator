# Day 4: Korean NLP with KoNLPy Lab
# Goal: Learn to process Korean text with morphological analysis

"""
INSTALLATION INSTRUCTIONS:
1. Install KoNLPy:
   pip install konlpy

2. Install Mecab (best Korean tokenizer):
   Mac/Linux:
   bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
   
   Windows:
   Follow instructions at: https://konlpy.org/en/latest/install/#windows

3. Test installation:
   python -c "from konlpy.tag import Mecab; print('Success!')"
"""

from konlpy.tag import Mecab, Okt, Kkma
from collections import Counter
import spacy

# For English comparison
nlp = spacy.load("en_core_web_sm")

def day4_intro():
    """
    Why Korean NLP is different from English
    """
    print("=" * 80)
    print("DAY 4: KOREAN NLP WITH KONLPY")
    print("=" * 80)
    
    print("\n1. WHY KOREAN NLP IS SPECIAL")
    print("-" * 80)
    print("""
    Korean has unique challenges that make NLP different:
    
    1. AGGLUTINATIVE LANGUAGE
       - Words formed by combining multiple morphemes
       - Example: 먹었습니다 = 먹 (eat) + 었 (past) + 습니다 (formal)
    
    2. NO SPACES IN COMPOUNDS
       - 한국어 (Korean language) is written as one word
       - Need to identify boundaries
    
    3. PARTICLES ATTACH TO WORDS
       - 은/는 (topic marker), 이/가 (subject marker), 을/를 (object marker)
       - Example: 나는 (I + topic marker)
    
    4. CONTEXTUAL MEANING
       - Words can have different meanings based on context
       - Particles indicate grammatical role
    
    5. WORD ORDER: SOV (Subject-Object-Verb)
       - English: "I eat rice" (SVO)
       - Korean: "나는 밥을 먹는다" (I rice eat - SOV)
    """)

def day4_tokenization_comparison():
    """
    Compare English and Korean tokenization
    """
    print("\n2. TOKENIZATION: ENGLISH VS KOREAN")
    print("-" * 80)
    
    # English tokenization
    english_text = "I eat rice."
    doc = nlp(english_text)
    
    print(f"ENGLISH: {english_text}")
    print(f"Tokens (space-based): {[token.text for token in doc]}")
    print("✓ Simple: spaces separate words\n")
    
    # Korean tokenization
    korean_text = "나는 밥을 먹는다."
    
    print(f"KOREAN: {korean_text}")
    print(f"Space-based tokens: {korean_text.split()}")
    print("✗ Problem: Doesn't separate particles or morphemes!")
    
    # Using Mecab for proper Korean tokenization
    mecab = Mecab()
    print(f"\nWith Mecab (morphological analysis):")
    print(f"Morphemes: {mecab.morphs(korean_text)}")
    print("✓ Better: Separates particles and verb stems!")
    
    # Show the breakdown
    print(f"\nDetailed breakdown:")
    for word, pos in mecab.pos(korean_text):
        print(f"  {word:<10} → {pos}")

def day4_korean_tokenizers():
    """
    Different Korean tokenizers and their characteristics
    """
    print("\n3. KOREAN TOKENIZER OPTIONS")
    print("-" * 80)
    
    text = "안녕하세요. 저는 한국어를 공부합니다."
    
    print(f"Text: {text}\n")
    
    # Mecab - Fastest and most accurate
    print("1. MECAB (Recommended - Fast & Accurate)")
    mecab = Mecab()
    print(f"   Morphemes: {mecab.morphs(text)}")
    print(f"   With POS: {mecab.pos(text)}")
    
    # Okt (formerly Twitter) - Good for informal text
    print("\n2. OKT (Good for social media/informal text)")
    okt = Okt()
    print(f"   Morphemes: {okt.morphs(text)}")
    print(f"   With POS: {okt.pos(text)}")
    
    # Kkma - Most detailed analysis
    print("\n3. KKMA (Most detailed, slowest)")
    kkma = Kkma()
    print(f"   Morphemes: {kkma.morphs(text)}")
    print(f"   Sentences: {kkma.sentences(text)}")

def day4_pos_tagging_korean():
    """
    Korean POS tagging
    """
    print("\n4. KOREAN POS TAGGING")
    print("-" * 80)
    
    text = "나는 학생입니다. 한국어를 공부해요."
    mecab = Mecab()
    
    print(f"Text: {text}\n")
    print(f"{'Morpheme':<15} {'POS Tag':<10} {'Meaning'}")
    print("-" * 50)
    
    # Common Korean POS tags
    pos_meanings = {
        'NNG': 'General Noun',
        'NNP': 'Proper Noun',
        'NNB': 'Bound Noun',
        'VV': 'Verb',
        'VA': 'Adjective',
        'MAG': 'General Adverb',
        'JKS': 'Subject Particle',
        'JKO': 'Object Particle',
        'JX': 'Auxiliary Particle',
        'JC': 'Conjunction Particle',
        'EP': 'Pre-final Ending',
        'EF': 'Final Ending',
        'EC': 'Connecting Ending',
        'ETM': 'Adnominal Ending',
        'XSV': 'Verb Suffix',
        'SF': 'Final Punctuation',
        'SP': 'Comma/Period',
    }
    
    for word, pos in mecab.pos(text):
        meaning = pos_meanings.get(pos, 'Other')
        print(f"{word:<15} {pos:<10} {meaning}")

def day4_particles():
    """
    Understanding Korean particles
    """
    print("\n5. KOREAN PARTICLES (조사)")
    print("-" * 80)
    
    print("""
    Particles are crucial in Korean - they mark grammatical roles:
    
    SUBJECT MARKERS:
      - 이/가 (i/ga) - subject marker
      - 은/는 (eun/neun) - topic marker
    
    OBJECT MARKERS:
      - 을/를 (eul/reul) - object marker
    
    LOCATION/DIRECTION:
      - 에 (e) - at, to, in
      - 에서 (eseo) - at, in (action location)
      - 으로/로 (euro/ro) - to, toward
    
    OTHER:
      - 의 (ui) - possessive (like 's in English)
      - 와/과 (wa/gwa) - and, with
      - 도 (do) - also, too
    """)
    
    examples = [
        ("나는 학생이다", "I am a student (topic: I)"),
        ("나가 했다", "I did it (emphasis on I)"),
        ("책을 읽는다", "read a book (book is object)"),
        ("학교에 간다", "go to school"),
        ("학교에서 공부한다", "study at school"),
    ]
    
    mecab = Mecab()
    
    print("\nExamples with particle identification:")
    for korean, english in examples:
        print(f"\n{korean} → {english}")
        tokens = mecab.pos(korean)
        particles = [word for word, pos in tokens if pos.startswith('J')]
        print(f"  Particles: {particles}")

def day4_verb_conjugation():
    """
    Korean verb conjugation analysis
    """
    print("\n6. KOREAN VERB CONJUGATION")
    print("-" * 80)
    
    print("""
    Korean verbs change based on:
    - Tense (past, present, future)
    - Politeness level (formal, informal, casual)
    - Mood (declarative, interrogative, imperative)
    
    Example: 먹다 (to eat)
    """)
    
    verb_forms = [
        ("먹는다", "eat (present, informal)"),
        ("먹어요", "eat (present, polite)"),
        ("먹습니다", "eat (present, formal)"),
        ("먹었다", "ate (past, informal)"),
        ("먹었어요", "ate (past, polite)"),
        ("먹었습니다", "ate (past, formal)"),
        ("먹을 것이다", "will eat (future)"),
    ]
    
    mecab = Mecab()
    
    print("\nVerb conjugation breakdown:")
    for korean, english in verb_forms:
        print(f"\n{korean} → {english}")
        morphs = mecab.pos(korean)
        print(f"  Morphemes: {morphs}")

def day4_translation_relevance():
    """
    How this applies to translation
    """
    print("\n7. APPLYING TO KOREAN-ENGLISH TRANSLATION")
    print("-" * 80)
    
    print("""
    For your noonchi-translator, understanding these concepts is critical:
    
    1. WORD ORDER TRANSFORMATION
       Korean: 나는 (I) 밥을 (rice) 먹는다 (eat) → SOV
       English: I eat rice → SVO
       Need to: Identify subject, object, verb and reorder
    
    2. PARTICLE REMOVAL/TRANSLATION
       Korean: 나는 (I + topic marker)
       English: "I" (no particle needed)
       Need to: Recognize and remove/translate particles appropriately
    
    3. VERB CONJUGATION MATCHING
       Korean: 먹었습니다 (ate, formal)
       English: "ate" (no formality marker)
       Need to: Extract tense, ignore politeness level
    
    4. CONTEXT PRESERVATION
       Korean particles give grammatical context
       English uses word order for context
       Need to: Convert particle-based context to word-order context
    """)
    
    # Example translation process
    print("\nExample translation process:")
    korean = "저는 어제 친구와 영화를 봤어요."
    english_target = "I watched a movie with my friend yesterday."
    
    mecab = Mecab()
    
    print(f"\nKorean: {korean}")
    print(f"Target English: {english_target}\n")
    
    print("Step 1: Morphological analysis")
    morphs = mecab.pos(korean)
    for word, pos in morphs:
        print(f"  {word:<10} {pos}")
    
    print("\nStep 2: Identify components")
    print("  Subject: 저 (I)")
    print("  Time: 어제 (yesterday)")
    print("  Object: 영화 (movie)")
    print("  Companion: 친구 (friend)")
    print("  Verb: 보다 (watch) + past tense")
    
    print("\nStep 3: Reorder for English (SVO + modifiers)")
    print("  I + watched + movie + with friend + yesterday")

def day4_practice():
    """
    Practice exercises
    """
    print("\n8. PRACTICE EXERCISES")
    print("-" * 80)
    
    mecab = Mecab()
    
    # Exercise 1: Sentence analysis
    print("\nExercise 1: Analyze this sentence")
    sent1 = "오늘은 날씨가 좋습니다."
    print(f"Sentence: {sent1} (Today the weather is good)")
    print(f"Morphemes: {mecab.morphs(sent1)}")
    print(f"POS tags: {mecab.pos(sent1)}")
    
    # Exercise 2: Find all particles
    print("\nExercise 2: Extract all particles")
    sent2 = "나는 학교에서 친구와 공부를 했다."
    particles = [word for word, pos in mecab.pos(sent2) if pos.startswith('J')]
    print(f"Sentence: {sent2}")
    print(f"Particles found: {particles}")
    
    # Exercise 3: Identify verb stems
    print("\nExercise 3: Find verb stems")
    sent3 = "밥을 먹고 영화를 봤어요."
    verbs = [word for word, pos in mecab.pos(sent3) if pos in ['VV', 'VA']]
    print(f"Sentence: {sent3}")
    print(f"Verb stems: {verbs}")

def day4_homework():
    """
    Homework assignments
    """
    print("\n9. HOMEWORK")
    print("-" * 80)
    
    print("""
    Complete these exercises:
    
    1. Tokenize these Korean sentences and identify all particles:
       - "저의 친구는 서울에 살고 있습니다."
       - "한국어를 배우는 것은 재미있어요."
    
    2. Compare the same sentence tokenized with Mecab, Okt, and Kkma.
       Which gives better results? Why?
    
    3. Create a function that:
       - Takes Korean text
       - Extracts all nouns
       - Extracts all verbs
       - Identifies the subject and object (using particles)
    
    4. Analyze verb conjugations:
       - Take the verb 가다 (to go)
       - Find different forms in sentences
       - Identify tense and politeness level
    
    5. Build a simple particle translator:
       - Input: Korean sentence with particles
       - Output: Identify what each particle means
       - Example: "나는" → "I (topic marker)"
    
    6. CHALLENGE: Create a basic word order converter
       - Input: Korean sentence (SOV)
       - Identify: Subject, Object, Verb
       - Output: English order (SVO)
    """)
    
    # TODO: Implement your solutions here
    
    def extract_nouns_verbs(text):
        """Extract nouns and verbs from Korean text"""
        # Your code here
        pass
    
    def identify_subject_object(text):
        """Use particles to identify subject and object"""
        # Your code here
        pass
    
    def analyze_verb_conjugation(verb_text):
        """Analyze tense and politeness of verb"""
        # Your code here
        pass

def day4_comparison_table():
    """
    Quick reference comparison
    """
    print("\n10. QUICK REFERENCE: ENGLISH VS KOREAN NLP")
    print("-" * 80)
    
    print(f"{'Feature':<25} {'English':<30} {'Korean'}")
    print("-" * 80)
    
    comparisons = [
        ("Tokenization", "Space-based (easy)", "Morpheme-based (complex)"),
        ("Word Order", "SVO", "SOV"),
        ("Particles", "None", "Essential (은/는, 이/가, 을/를)"),
        ("Verb Conjugation", "Simple (eat, ate, eaten)", "Complex (politeness + tense)"),
        ("Tool", "spaCy", "KoNLPy (Mecab, Okt, Kkma)"),
        ("POS Tags", "Universal (17 tags)", "Korean-specific (40+ tags)"),
        ("Main Challenge", "Word sense", "Morpheme boundaries"),
    ]
    
    for feature, english, korean in comparisons:
        print(f"{feature:<25} {english:<30} {korean}")

if __name__ == "__main__":
    day4_intro()
    day4_tokenization_comparison()
    day4_korean_tokenizers()
    day4_pos_tagging_korean()
    day4_particles()
    day4_verb_conjugation()
    day4_translation_relevance()
    day4_practice()
    day4_homework()
    day4_comparison_table()
    
    print("\n" + "=" * 80)
    print("DAY 4 COMPLETE!")
    print("=" * 80)
    print("""
    You've learned:
    ✓ Why Korean NLP is different from English
    ✓ How to use KoNLPy (Mecab, Okt, Kkma)
    ✓ Korean morphological analysis
    ✓ Korean particles and their roles
    ✓ Korean verb conjugation
    ✓ How Korean NLP applies to translation
    
    Next up: Day 5 - Building a Simple Korean-English Translator
    """)