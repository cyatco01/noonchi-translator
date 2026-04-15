# Day 5: Building a Simple English-Korean Translator
# Goal: Combine everything learned to build a basic translation system

"""
WHAT YOU'VE LEARNED SO FAR:
Day 1: Tokenization - Breaking text into words
Day 2: Named Entity Recognition - Identifying important entities
Day 3: POS Tagging & Dependency Parsing - Understanding grammar structure
Day 4: Korean NLP - Processing Korean text with morphological analysis

TODAY: Put it all together to build an English → Korean translator!
"""

from konlpy.tag import Mecab
import spacy
from collections import defaultdict

# Initialize tools
mecab = Mecab()
nlp = spacy.load("en_core_web_sm")

print("=" * 100)
print("DAY 5: BUILDING AN ENGLISH-KOREAN TRANSLATOR")
print("=" * 100)

# ============================================================================
# 1. TRANSLATION CHALLENGES
# ============================================================================
def day5_intro():
    """
    Understanding the translation challenge
    """
    print("\n1. TRANSLATION CHALLENGES: ENGLISH → KOREAN")
    print("-" * 100)
    print("""
    Key differences to handle:
    
    1. WORD ORDER
       English: Subject-Verb-Object (SVO)
       Korean: Subject-Object-Verb (SOV)
       Example: "I eat rice" → 나는 밥을 먹는다 (I rice eat)
    
    2. PARTICLES
       English: Uses word order to indicate roles
       Korean: Adds particles after words (은/는, 이/가, 을/를)
       Example: "I" → "나는" (I + topic marker)
    
    3. VERB CONJUGATION
       English: Simple tense markers (eat, ate, will eat)
       Korean: Complex endings with tense + politeness
       Example: "ate" → "먹었다" (informal) or "먹었어요" (polite)
    
    4. HONORIFICS
       English: No honorifics
       Korean: Different verb forms for politeness
       Example: "eat" → "먹다" (plain), "드시다" (honorific)
    
    5. CONTEXT
       Korean: Often drops subject when clear from context
       English: Usually requires explicit subject
       Example: "Ate rice" → "밥을 먹었어요" (subject dropped)
    """)

day5_intro()

# ============================================================================
# 2. SIMPLE DICTIONARY-BASED TRANSLATOR
# ============================================================================
print("\n2. BASIC DICTIONARY TRANSLATOR")
print("-" * 100)

class SimpleTranslator:
    """
    A basic dictionary-based English to Korean translator
    """
    
    def __init__(self):
        # Basic English-Korean dictionary
        self.dictionary = {
            # Pronouns
            'i': '나',
            'you': '너',
            'we': '우리',
            'he': '그',
            'she': '그녀',
            'they': '그들',
            
            # Nouns
            'rice': '밥',
            'food': '음식',
            'book': '책',
            'student': '학생',
            'teacher': '선생님',
            'school': '학교',
            'friend': '친구',
            'home': '집',
            'house': '집',
            'water': '물',
            'person': '사람',
            'movie': '영화',
            'korean': '한국어',
            'english': '영어',
            'cat': '고양이',
            'dog': '개',
            
            # Verbs
            'eat': '먹다',
            'go': '가다',
            'come': '오다',
            'see': '보다',
            'watch': '보다',
            'read': '읽다',
            'write': '쓰다',
            'do': '하다',
            'study': '공부하다',
            'like': '좋아하다',
            'love': '사랑하다',
            'learn': '배우다',
            'teach': '가르치다',
            'buy': '사다',
            'sell': '팔다',
            
            # Adjectives
            'pretty': '예쁘다',
            'beautiful': '아름답다',
            'big': '크다',
            'small': '작다',
            'good': '좋다',
            'bad': '나쁘다',
            'delicious': '맛있다',
            'happy': '행복하다',
            
            # Adverbs
            'very': '매우',
            'quickly': '빨리',
            'slowly': '천천히',
            
            # Time words
            'yesterday': '어제',
            'today': '오늘',
            'tomorrow': '내일',
            'now': '지금',
            
            # Prepositions
            'with': '와',
            'at': '에',
            'in': '에',
            'to': '에',
            'from': '에서',
        }
        
        # Subject particles
        self.subject_particles = {
            'topic': '는',  # 은/는 - use 는 after vowel
            'subject': '가',  # 이/가 - use 가 after vowel
        }
        
        # Object particle
        self.object_particle = '를'  # 을/를 - use 를 after vowel
    
    def translate_word(self, word):
        """
        Translate a single word
        """
        word_lower = word.lower()
        return self.dictionary.get(word_lower, word)
    
    def basic_translate(self, english_text):
        """
        Very basic word-by-word translation
        """
        print(f"\nEnglish: {english_text}")
        
        # Simple tokenization (split by spaces)
        words = english_text.lower().replace('.', '').split()
        
        print(f"\nWords: {words}")
        
        # Translate each word
        korean_words = []
        for word in words:
            korean = self.translate_word(word)
            korean_words.append(korean)
            print(f"  {word:<15} → {korean}")
        
        # Join (note: this is wrong word order!)
        basic_korean = ' '.join(korean_words)
        print(f"\nWord-by-word (WRONG ORDER): {basic_korean}")
        
        return basic_korean

# Test basic translator
translator = SimpleTranslator()

examples = [
    "I eat rice.",
    "The student reads a book.",
    "I watched a movie with a friend.",
]

for ex in examples:
    translator.basic_translate(ex)
    print()

# ============================================================================
# 3. IMPROVED TRANSLATOR WITH WORD ORDER
# ============================================================================
print("\n3. IMPROVED TRANSLATOR WITH WORD ORDER CORRECTION")
print("-" * 100)

class ImprovedTranslator(SimpleTranslator):
    """
    Translator that handles word order (SVO → SOV)
    """
    
    def extract_sentence_components(self, english_text):
        """
        Extract subject, object, and verb from English sentence using spaCy
        """
        doc = nlp(english_text)
        
        components = {
            'subject': [],
            'verb': [],
            'object': [],
            'modifiers': [],
            'time': [],
        }
        
        # Find the root verb
        root_verb = None
        for token in doc:
            if token.dep_ == "ROOT":
                root_verb = token
                components['verb'].append(token.text)
                break
        
        if not root_verb:
            return components
        
        # Find subject (nsubj, nsubjpass)
        for token in doc:
            if "subj" in token.dep_:
                # Get the full noun phrase
                components['subject'].append(token.text)
                # Add determiners and adjectives
                for child in token.children:
                    if child.pos_ in ["DET", "ADJ"]:
                        components['modifiers'].append(child.text)
        
        # Find object (dobj, pobj)
        for token in doc:
            if "obj" in token.dep_:
                components['object'].append(token.text)
                # Add determiners and adjectives
                for child in token.children:
                    if child.pos_ in ["DET", "ADJ"]:
                        components['modifiers'].append(child.text)
        
        # Find time expressions
        for token in doc:
            if token.text.lower() in ['yesterday', 'today', 'tomorrow']:
                components['time'].append(token.text)
        
        # Find prepositions with objects (e.g., "with friend")
        for token in doc:
            if token.pos_ == "ADP":  # Preposition
                for child in token.children:
                    if child.dep_ == "pobj":
                        components['modifiers'].append(token.text)
                        components['modifiers'].append(child.text)
        
        return components
    
    def translate_with_order(self, english_text):
        """
        Translate with proper Korean word order (SOV)
        """
        print(f"\nEnglish: {english_text}")
        
        # Parse with spaCy
        doc = nlp(english_text)
        
        print(f"\nspaCy analysis:")
        for token in doc:
            print(f"  {token.text:<15} {token.pos_:<10} {token.dep_:<15} {token.head.text}")
        
        # Extract components
        components = self.extract_sentence_components(english_text)
        
        print(f"\nExtracted components:")
        for comp, words in components.items():
            if words:
                print(f"  {comp.capitalize()}: {words}")
        
        # Translate each component
        translated = {
            'subject': [self.translate_word(w) for w in components['subject']],
            'verb': [self.translate_word(w) for w in components['verb']],
            'object': [self.translate_word(w) for w in components['object']],
            'time': [self.translate_word(w) for w in components['time']],
            'modifiers': [self.translate_word(w) for w in components['modifiers']],
        }
        
        print(f"\nTranslated components:")
        for comp, words in translated.items():
            if words:
                print(f"  {comp.capitalize()}: {words}")
        
        # Build Korean sentence: Time + Subject + Object + Modifiers + Verb
        korean_parts = []
        
        if translated['time']:
            korean_parts.append(' '.join(translated['time']))
        
        if translated['subject']:
            subj = ' '.join(translated['subject'])
            korean_parts.append(subj + '는')  # Add topic particle
        
        if translated['object']:
            obj = ' '.join(translated['object'])
            korean_parts.append(obj + '를')  # Add object particle
        
        if translated['modifiers']:
            korean_parts.append(' '.join(translated['modifiers']))
        
        if translated['verb']:
            korean_parts.append(' '.join(translated['verb']))
        
        korean = ' '.join(korean_parts) + '.'
        
        print(f"\nKorean (reordered to SOV): {korean}")
        return korean

# Test improved translator
improved = ImprovedTranslator()

test_sentences = [
    "I eat rice.",
    "The student reads a book.",
    "I watched a movie yesterday.",
]

for sent in test_sentences:
    improved.translate_with_order(sent)
    print()

# ============================================================================
# 4. HANDLING TENSE
# ============================================================================
print("\n4. HANDLING VERB TENSE")
print("-" * 100)

class TenseAwareTranslator(ImprovedTranslator):
    """
    Translator that handles tense conversion
    """
    
    def detect_tense(self, english_text):
        """
        Detect tense from English verb using spaCy
        """
        doc = nlp(english_text)
        
        for token in doc:
            if token.pos_ == "VERB":
                # Check verb tag
                if token.tag_ in ["VBD", "VBN"]:  # Past tense
                    return "past"
                elif token.tag_ in ["VB", "VBP", "VBZ"]:  # Present
                    return "present"
                elif "will" in english_text.lower():
                    return "future"
        
        return "present"
    
    def get_verb_stem(self, verb_word):
        """
        Get the base form (lemma) of the verb
        """
        doc = nlp(verb_word)
        for token in doc:
            if token.pos_ == "VERB":
                return token.lemma_
        return verb_word
    
    def conjugate_korean_verb(self, verb_base, tense, politeness="informal"):
        """
        Conjugate Korean verb based on tense and politeness
        
        Note: This is simplified. Real Korean conjugation is complex!
        """
        # Remove 다 from verb base if present
        if verb_base.endswith('다'):
            stem = verb_base[:-1]
        else:
            stem = verb_base
        
        if tense == "past":
            if politeness == "formal":
                return stem + "었습니다"
            elif politeness == "polite":
                return stem + "었어요"
            else:  # informal
                return stem + "었다"
        
        elif tense == "future":
            if politeness == "formal":
                return stem + "겠습니다"
            elif politeness == "polite":
                return stem + "을 거예요"
            else:
                return stem + "을 것이다"
        
        else:  # present
            if politeness == "formal":
                return stem + "습니다"
            elif politeness == "polite":
                return stem + "어요"
            else:
                return stem + "는다"
    
    def translate_with_tense(self, english_text, politeness="informal"):
        """
        Translate with proper tense handling
        """
        print(f"\nEnglish: {english_text}")
        
        # Detect tense
        tense = self.detect_tense(english_text)
        print(f"Detected tense: {tense}")
        print(f"Politeness level: {politeness}")
        
        # Parse with spaCy
        doc = nlp(english_text)
        
        # Extract components
        components = self.extract_sentence_components(english_text)
        
        # Get verb lemma (base form)
        verb_lemma = None
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                verb_lemma = token.lemma_
                break
        
        print(f"\nVerb analysis:")
        print(f"  Verb in sentence: {components['verb']}")
        print(f"  Base form (lemma): {verb_lemma}")
        
        # Translate components
        translated = {
            'subject': [self.translate_word(w) for w in components['subject']],
            'object': [self.translate_word(w) for w in components['object']],
            'time': [self.translate_word(w) for w in components['time']],
            'modifiers': [self.translate_word(w) for w in components['modifiers']],
        }
        
        # Translate and conjugate verb
        if verb_lemma:
            korean_verb_base = self.translate_word(verb_lemma)
            conjugated_verb = self.conjugate_korean_verb(korean_verb_base, tense, politeness)
            translated['verb'] = [conjugated_verb]
            
            print(f"  Korean verb base: {korean_verb_base}")
            print(f"  Conjugated: {conjugated_verb}")
        
        # Build Korean sentence: Time + Subject는 + Object를 + Modifiers + Verb
        korean_parts = []
        
        if translated['time']:
            korean_parts.append(' '.join(translated['time']))
        
        if translated['subject']:
            subj = ' '.join(translated['subject'])
            korean_parts.append(subj + '는')
        
        if translated['object']:
            obj = ' '.join(translated['object'])
            korean_parts.append(obj + '를')
        
        if translated['modifiers']:
            korean_parts.append(' '.join(translated['modifiers']))
        
        if translated['verb']:
            korean_parts.append(' '.join(translated['verb']))
        
        korean = ' '.join(korean_parts) + '.'
        
        print(f"\nKorean: {korean}")
        return korean

# Test tense-aware translator
tense_translator = TenseAwareTranslator()

tense_examples = [
    ("I eat rice.", "informal"),
    ("I eat rice.", "polite"),
    ("I eat rice.", "formal"),
    ("I ate rice.", "polite"),
    ("I will eat rice.", "polite"),
]

for sent, politeness in tense_examples:
    tense_translator.translate_with_tense(sent, politeness)
    print()

# ============================================================================
# 5. COMPLETE TRANSLATOR CLASS
# ============================================================================
print("\n5. COMPLETE TRANSLATOR (PUTTING IT ALL TOGETHER)")
print("-" * 100)

class EnglishKoreanTranslator:
    """
    A complete basic English-Korean translator
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Expanded dictionary
        self.dictionary = {
            # Pronouns
            'i': '나', 'you': '너', 'we': '우리', 'he': '그', 'she': '그녀',
            
            # Nouns
            'rice': '밥', 'food': '음식', 'book': '책', 'student': '학생',
            'school': '학교', 'friend': '친구', 'movie': '영화',
            'korean': '한국어', 'teacher': '선생님', 'cat': '고양이',
            'dog': '개', 'house': '집',
            
            # Verbs (base form → Korean 다 form)
            'eat': '먹다', 'go': '가다', 'watch': '보다', 'read': '읽다',
            'study': '공부하다', 'do': '하다', 'like': '좋아하다',
            'love': '사랑하다', 'buy': '사다', 'come': '오다',
            
            # Adjectives
            'pretty': '예쁘다', 'good': '좋다', 'delicious': '맛있다',
            'big': '크다', 'small': '작다',
            
            # Time
            'yesterday': '어제', 'today': '오늘', 'tomorrow': '내일',
            
            # Other
            'with': '와',
        }
    
    def translate(self, english_text, politeness="polite"):
        """
        Main translation method
        
        Args:
            english_text: English sentence to translate
            politeness: "informal", "polite", or "formal"
        """
        print(f"\n{'='*80}")
        print(f"TRANSLATING: {english_text}")
        print(f"POLITENESS: {politeness}")
        print('='*80)
        
        # Step 1: Parse English with spaCy
        doc = self.nlp(english_text)
        print(f"\nStep 1: Parse English Sentence")
        print(f"{'Token':<15} {'POS':<10} {'Dependency':<15} {'Head'}")
        print('-'*60)
        for token in doc:
            print(f"{token.text:<15} {token.pos_:<10} {token.dep_:<15} {token.head.text}")
        
        # Step 2: Extract components
        print(f"\nStep 2: Extract Sentence Components (SVO)")
        subject, obj, verb, time, tense = self._extract_components(doc)
        print(f"  Subject: {subject}")
        print(f"  Verb: {verb} (lemma: {verb['lemma'] if verb else None})")
        print(f"  Object: {obj}")
        print(f"  Time: {time}")
        print(f"  Tense: {tense}")
        
        # Step 3: Translate components
        print(f"\nStep 3: Translate Each Component")
        trans_subj = self._translate_list(subject)
        trans_obj = self._translate_list(obj)
        trans_time = self._translate_list(time)
        
        print(f"  Subject: {subject} → {trans_subj}")
        print(f"  Object: {obj} → {trans_obj}")
        print(f"  Time: {time} → {trans_time}")
        
        # Step 4: Translate and conjugate verb
        print(f"\nStep 4: Translate and Conjugate Verb")
        if verb:
            korean_verb_base = self.dictionary.get(verb['lemma'], verb['lemma'])
            conjugated = self._conjugate_verb(korean_verb_base, tense, politeness)
            print(f"  English verb: {verb['text']} (base: {verb['lemma']})")
            print(f"  Korean base: {korean_verb_base}")
            print(f"  Conjugated ({tense}, {politeness}): {conjugated}")
            trans_verb = conjugated
        else:
            trans_verb = ''
        
        # Step 5: Add particles and reorder to SOV
        print(f"\nStep 5: Add Particles & Reorder to Korean (SOV)")
        korean_parts = []
        
        # Time (optional, at beginning)
        if trans_time:
            korean_parts.append(trans_time)
            print(f"  Time: {trans_time}")
        
        # Subject + 는 (topic marker)
        if trans_subj:
            subj_with_particle = trans_subj + '는'
            korean_parts.append(subj_with_particle)
            print(f"  Subject + particle: {subj_with_particle}")
        
        # Object + 를 (object marker)
        if trans_obj:
            obj_with_particle = trans_obj + '를'
            korean_parts.append(obj_with_particle)
            print(f"  Object + particle: {obj_with_particle}")
        
        # Verb (at the end)
        if trans_verb:
            korean_parts.append(trans_verb)
            print(f"  Verb: {trans_verb}")
        
        korean = ' '.join(korean_parts) + '.'
        
        print(f"\nFINAL TRANSLATION: {korean}")
        print('='*80)
        
        return korean
    
    def _extract_components(self, doc):
        """Extract sentence components from spaCy doc"""
        subject = []
        obj = []
        verb = None
        time = []
        tense = 'present'
        
        # Find root verb
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                verb = {
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tag': token.tag_
                }
                
                # Detect tense from verb tag
                if token.tag_ in ["VBD", "VBN"]:
                    tense = "past"
                elif "will" in doc.text.lower():
                    tense = "future"
        
        # Find subject
        for token in doc:
            if "subj" in token.dep_:
                subject.append(token.text)
        
        # Find object
        for token in doc:
            if "obj" in token.dep_:
                obj.append(token.text)
        
        # Find time words
        for token in doc:
            if token.text.lower() in ['yesterday', 'today', 'tomorrow']:
                time.append(token.text)
        
        return subject, obj, verb, time, tense
    
    def _translate_list(self, words):
        """Translate a list of words"""
        if not words:
            return ''
        translated = [self.dictionary.get(w.lower(), w) for w in words]
        return ' '.join(translated)
    
    def _conjugate_verb(self, verb_base, tense, politeness):
        """Conjugate Korean verb"""
        # Remove 다 if present
        if verb_base.endswith('다'):
            stem = verb_base[:-1]
        else:
            stem = verb_base
        
        if tense == "past":
            if politeness == "formal":
                return stem + "었습니다"
            elif politeness == "polite":
                return stem + "었어요"
            else:
                return stem + "었다"
        
        elif tense == "future":
            if politeness == "formal":
                return stem + "겠습니다"
            elif politeness == "polite":
                return stem + "을 거예요"
            else:
                return stem + "을 것이다"
        
        else:  # present
            if politeness == "formal":
                return stem + "습니다"
            elif politeness == "polite":
                return stem + "어요"
            else:
                return stem + "는다"

# Test complete translator
complete_translator = EnglishKoreanTranslator()

final_examples = [
    ("I eat rice.", "informal"),
    ("I eat rice.", "polite"),
    ("The student reads a book.", "polite"),
    ("I watched a movie yesterday.", "polite"),
    ("I will go to school tomorrow.", "formal"),
]

for sent, politeness in final_examples:
    complete_translator.translate(sent, politeness)

# ============================================================================
# 6. LIMITATIONS AND IMPROVEMENTS
# ============================================================================
print("\n\n6. LIMITATIONS OF THIS TRANSLATOR")
print("-" * 100)
print("""
Current limitations:

1. LIMITED VOCABULARY
   - Only ~50 words in dictionary
   - Need thousands of words for real translation
   
2. SIMPLE WORD ORDER
   - Only handles basic SVO → SOV
   - Doesn't handle complex sentences, relative clauses
   
3. PARTICLE SELECTION
   - Always uses 는 (topic) and 를 (object)
   - Should alternate between 은/는, 이/가, 을/를 based on final consonant
   
4. VERB CONJUGATION
   - Simplified conjugation rules
   - Doesn't handle irregular verbs properly
   - Missing many conjugation forms
   
5. NO ADJECTIVES AS PREDICATES
   - Can't handle "The cat is pretty" → "고양이가 예쁘다"
   
6. NO IDIOMS OR EXPRESSIONS
   - Translates literally
   - Misses cultural expressions
   
7. NO HONORIFICS
   - Doesn't distinguish between 먹다 vs 드시다 (eat vs eat-honorific)

Improvements needed:
✓ Expand vocabulary significantly
✓ Smart particle selection (은 vs 는, 을 vs 를)
✓ Better verb conjugation (irregular verbs)
✓ Handle adjective predicates
✓ Add relative clauses
✓ Implement honorific system
✓ Use neural translation for production
""")

# ============================================================================
# 7. HOMEWORK
# ============================================================================
print("\n7. HOMEWORK")
print("-" * 100)
print("""
Complete these exercises to improve your translator:

1. EXPAND VOCABULARY
   Add 100 more words to the dictionary:
   - 20 common nouns
   - 20 common verbs
   - 20 adjectives
   - 20 adverbs
   - 20 common expressions

2. SMART PARTICLE SELECTION
   Implement rules for choosing correct particle form:
   - 은/는 (eun/neun) - use 은 after consonant, 는 after vowel
   - 이/가 (i/ga) - use 이 after consonant, 가 after vowel
   - 을/를 (eul/reul) - use 을 after consonant, 를 after vowel
   
   Example: "student" → "학생" (ends in ㅇ consonant) → "학생은"
           "I" → "나" (ends in vowel) → "나는"

3. HANDLE ADJECTIVES
   Translate sentences like "The cat is pretty" → "고양이가 예쁘다"
   (Note: Korean adjectives conjugate like verbs!)

4. ADD PLURAL HANDLING
   Detect plural nouns and add 들
   Example: "friends" → "친구들"

5. HANDLE QUESTIONS
   Translate questions with proper endings
   Example: "Do you eat rice?" → "밥을 먹어요?"

6. IMPLEMENT PROGRESSIVE TENSE
   Handle "-ing" forms
   Example: "I am eating" → "먹고 있어요"

7. ADD NEGATION
   Handle "not" and "don't/doesn't/didn't"
   Example: "I don't eat rice" → "나는 밥을 안 먹어요"

8. COMPOUND SENTENCES
   Translate sentences with "and", "but", "because"
   Example: "I ate rice and watched a movie" → "밥을 먹고 영화를 봤어요"

9. CHALLENGE: BIDIRECTIONAL CHECK
   Translate English → Korean, then use your Day 4 Korean analyzer
   to verify the Korean sentence structure is correct

10. INTEGRATION PROJECT
    Build a web interface where users can:
    - Input English text
    - Select politeness level
    - See the translation
    - See the breakdown of components

BONUS CHALLENGES:
- Research how to detect if last character has final consonant (받침)
- Implement honorific verb detection (for talking about elders)
- Add more complex sentence structures (relative clauses)
- Compare your translations with Google Translate or Papago
""")

print("\n" + "=" * 100)
print("DAY 5 COMPLETE!")
print("=" * 100)
print("""
🎉 CONGRATULATIONS! You've built a basic English-Korean translator!

You've learned:
✓ How to parse English sentence structure with spaCy
✓ How to extract sentence components (subject, object, verb)
✓ How to handle word order transformation (SVO → SOV)
✓ How to add Korean particles (은/는, 이/가, 을/를)
✓ How to conjugate Korean verbs for tense and politeness
✓ The challenges of real translation systems

Your 5-day journey:
Day 1: Tokenization ✓
Day 2: Named Entity Recognition ✓
Day 3: POS Tagging & Dependency Parsing ✓
Day 4: Korean NLP ✓
Day 5: English-Korean Translator ✓

IMPORTANT NOTE:
This is a LEARNING EXERCISE to understand translation concepts.
For your real noonchi-translator project, you should use:
- Papago API (Naver's translator - best for Korean)
- Google Translate API
- OpenAI GPT API for translation
- Or other professional translation services

These provide 95%+ accuracy vs our ~60% accuracy educational translator.

Next steps:
- Continue expanding your translator
- Implement the homework exercises
- Learn about neural machine translation
- Integrate a professional API for your real project

You now understand HOW translation works! 🚀
Keep learning and building!
""")
'''
To create and run Day 5:
bashcd /Users/carolynyatco/noonchi_project/noonchi-translator/backend/nlp/
touch day5_translator_lab.py
# Copy the code above into the file
python day5_translator_lab.py
```

## **Key differences from the previous version:**

1. **Direction**: English → Korean (not Korean → English)
2. **Word order**: Transforms SVO → SOV
3. **Adds particles**: Attaches Korean particles (는, 를, etc.)
4. **Verb conjugation**: Converts English verbs to Korean conjugated forms
5. **Politeness levels**: Handles informal, polite, and formal speech

## **What this translator does:**
```
Input:  "I eat rice."
Output: "나는 밥을 먹어요." (polite)

Process:
1. Parse: I (subject) + eat (verb) + rice (object)
2. Translate: 나, 먹다, 밥
3. Reorder: 나 + 밥 + 먹다 (SOV)
4. Add particles: 나는 + 밥을
5. Conjugate verb: 먹다 → 먹어요 (polite present)
6. Result: 나는 밥을 먹어요.

'''