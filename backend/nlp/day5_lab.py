# Day 5: Building a Simple Korean-English Translator
# Goal: Combine everything learned to build a basic translation system

"""
WHAT YOU'VE LEARNED SO FAR:
Day 1: Tokenization - Breaking text into words
Day 2: Named Entity Recognition - Identifying important entities
Day 3: POS Tagging & Dependency Parsing - Understanding grammar structure
Day 4: Korean NLP - Processing Korean text with morphological analysis

TODAY: Put it all together to build a translator!
"""

from konlpy.tag import Mecab
import spacy
from collections import defaultdict

# Initialize tools
mecab = Mecab()
nlp = spacy.load("en_core_web_sm")

print("=" * 100)
print("DAY 5: BUILDING A KOREAN-ENGLISH TRANSLATOR")
print("=" * 100)

# ============================================================================
# 1. TRANSLATION CHALLENGES
# ============================================================================
def day5_intro():
    """
    Understanding the translation challenge
    """
    print("\n1. TRANSLATION CHALLENGES: KOREAN → ENGLISH")
    print("-" * 100)
    print("""
    Key differences to handle:
    
    1. WORD ORDER
       Korean: Subject-Object-Verb (SOV)
       English: Subject-Verb-Object (SVO)
       Example: 나는 밥을 먹는다 (I rice eat) → "I eat rice"
    
    2. PARTICLES
       Korean: Uses particles to mark grammatical roles (은/는, 이/가, 을/를)
       English: Uses word order
       Example: 나는 (I + topic marker) → "I"
    
    3. VERB CONJUGATION
       Korean: Complex endings with tense + politeness
       English: Simple tense markers
       Example: 먹었습니다 (ate + formal) → "ate"
    
    4. DROPPED SUBJECTS
       Korean: Often omits subject when context is clear
       English: Usually requires explicit subject
       Example: 밥을 먹었어요 (ate rice) → "I ate rice" or "He/She ate rice"
    
    5. HONORIFICS
       Korean: Built into verb conjugations
       English: No equivalent
       Example: 드셨어요 (ate - honorific) → "ate" (lose politeness info)
    """)

day5_intro()

# ============================================================================
# 2. SIMPLE DICTIONARY-BASED TRANSLATOR
# ============================================================================
print("\n2. BASIC DICTIONARY TRANSLATOR")
print("-" * 100)

class SimpleTranslator:
    """
    A basic dictionary-based Korean to English translator
    """
    
    def __init__(self):
        # Basic Korean-English dictionary
        self.dictionary = {
            # Pronouns
            '나': 'I',
            '저': 'I',  # humble
            '너': 'you',
            '우리': 'we',
            '그': 'he',
            '그녀': 'she',
            
            # Nouns
            '밥': 'rice',
            '음식': 'food',
            '책': 'book',
            '학생': 'student',
            '선생님': 'teacher',
            '학교': 'school',
            '친구': 'friend',
            '집': 'home',
            '물': 'water',
            '사람': 'person',
            '영화': 'movie',
            '한국어': 'Korean',
            '영어': 'English',
            
            # Verbs (stems)
            '먹': 'eat',
            '가': 'go',
            '오': 'come',
            '보': 'see/watch',
            '읽': 'read',
            '쓰': 'write',
            '하': 'do',
            '공부': 'study',
            '좋아하': 'like',
            '사랑하': 'love',
            '배우': 'learn',
            
            # Adjectives
            '예쁘': 'pretty',
            '크': 'big',
            '작': 'small',
            '좋': 'good',
            '나쁘': 'bad',
            '맛있': 'delicious',
            
            # Adverbs
            '매우': 'very',
            '아주': 'very',
            '빨리': 'quickly',
            '천천히': 'slowly',
            '어제': 'yesterday',
            '오늘': 'today',
            '내일': 'tomorrow',
            
            # Time words
            '어제': 'yesterday',
            '오늘': 'today',
            '내일': 'tomorrow',
            '지금': 'now',
        }
        
        # Particle meanings (for reference, usually dropped in English)
        self.particles = {
            '은': 'topic',
            '는': 'topic',
            '이': 'subject',
            '가': 'subject',
            '을': 'object',
            '를': 'object',
            '와': 'with',
            '과': 'with',
            '의': "'s",
            '에': 'to/at',
            '에서': 'at/from',
            '도': 'also',
            '만': 'only',
        }
        
        # Tense markers
        self.tense_markers = {
            '었': 'past',
            '았': 'past',
            '겠': 'future/will',
        }
    
    def translate_word(self, word, pos):
        """
        Translate a single word based on its POS tag
        """
        # Check dictionary
        if word in self.dictionary:
            return self.dictionary[word]
        
        # Handle particles (usually dropped in English)
        if pos.startswith('J'):
            if word == '의':
                return "'s"
            return ''  # Most particles are dropped
        
        # Handle verb endings (dropped in English, but note tense)
        if pos in ['EP', 'EF', 'EC']:
            if word in self.tense_markers:
                return f"[{self.tense_markers[word]}]"
            return ''
        
        # Handle punctuation
        if pos.startswith('S'):
            return word
        
        # Unknown word - return as is
        return word
    
    def basic_translate(self, korean_text):
        """
        Very basic word-by-word translation
        """
        print(f"\nKorean: {korean_text}")
        
        # Analyze with Mecab
        morphs = mecab.pos(korean_text)
        
        print(f"\nMorphological analysis:")
        for word, pos in morphs:
            translation = self.translate_word(word, pos)
            print(f"  {word:<10} ({pos:<6}) → {translation}")
        
        # Simple word-by-word translation
        translations = []
        for word, pos in morphs:
            trans = self.translate_word(word, pos)
            if trans and not trans.startswith('['):  # Skip tense markers for now
                translations.append(trans)
        
        word_by_word = ' '.join(translations)
        print(f"\nWord-by-word: {word_by_word}")
        
        return word_by_word

# Test basic translator
translator = SimpleTranslator()

examples = [
    "나는 밥을 먹는다.",
    "학생이 책을 읽는다.",
    "친구와 영화를 봤어요.",
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
    Translator that handles word order (SOV → SVO)
    """
    
    def extract_sentence_components(self, korean_text):
        """
        Extract subject, object, and verb from Korean sentence
        """
        morphs = mecab.pos(korean_text)
        
        components = {
            'subject': [],
            'object': [],
            'verb': [],
            'modifiers': [],
            'time': [],
        }
        
        current_component = None
        
        for i, (word, pos) in enumerate(morphs):
            # Identify component based on particle
            if pos == 'JKS' or (pos == 'JX' and word in ['은', '는']):
                # Previous word(s) are subject
                current_component = 'subject'
            elif pos == 'JKO':
                # Previous word(s) are object
                current_component = 'object'
            elif pos in ['VV', 'VA']:
                # Verb stem
                current_component = 'verb'
            
            # Add word to appropriate component
            if pos.startswith('N'):  # Noun
                if current_component and current_component != 'verb':
                    components[current_component].append(word)
                else:
                    components['modifiers'].append(word)
            elif pos in ['VV', 'VA']:  # Verb/Adjective
                components['verb'].append(word)
            elif pos == 'MAG':  # Adverb (time words)
                components['time'].append(word)
            elif pos.startswith('M'):  # Other modifiers
                components['modifiers'].append(word)
        
        return components
    
    def translate_with_order(self, korean_text):
        """
        Translate with proper English word order (SVO)
        """
        print(f"\nKorean: {korean_text}")
        
        # Extract components
        components = self.extract_sentence_components(korean_text)
        
        print(f"\nExtracted components:")
        for comp, words in components.items():
            if words:
                print(f"  {comp.capitalize()}: {words}")
        
        # Translate each component
        translated = {
            'subject': [self.dictionary.get(w, w) for w in components['subject']],
            'verb': [self.dictionary.get(w, w) for w in components['verb']],
            'object': [self.dictionary.get(w, w) for w in components['object']],
            'time': [self.dictionary.get(w, w) for w in components['time']],
            'modifiers': [self.dictionary.get(w, w) for w in components['modifiers']],
        }
        
        # Build English sentence: Subject + Verb + Object + Modifiers + Time
        english_parts = []
        
        if translated['subject']:
            english_parts.append(' '.join(translated['subject']))
        
        if translated['verb']:
            english_parts.append(' '.join(translated['verb']))
        
        if translated['object']:
            english_parts.append(' '.join(translated['object']))
        
        if translated['modifiers']:
            english_parts.append(' '.join(translated['modifiers']))
        
        if translated['time']:
            english_parts.append(' '.join(translated['time']))
        
        english = ' '.join(english_parts) + '.'
        
        print(f"\nEnglish (reordered): {english}")
        return english

# Test improved translator
improved = ImprovedTranslator()

test_sentences = [
    "나는 밥을 먹는다.",
    "학생이 책을 읽는다.",
    "친구와 영화를 봤어요.",
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
    Translator that handles tense markers
    """
    
    def detect_tense(self, korean_text):
        """
        Detect tense from verb endings
        """
        morphs = mecab.pos(korean_text)
        
        for word, pos in morphs:
            if pos == 'EP':  # Pre-final ending (tense marker)
                if word in ['었', '았']:
                    return 'past'
                elif word == '겠':
                    return 'future'
        
        return 'present'
    
    def conjugate_verb(self, verb_stem, tense):
        """
        Simple verb conjugation for English
        """
        # This is simplified - real English conjugation is more complex
        if tense == 'past':
            # Simple past (very simplified)
            if verb_stem == 'eat':
                return 'ate'
            elif verb_stem == 'go':
                return 'went'
            elif verb_stem == 'see':
                return 'saw'
            elif verb_stem == 'read':
                 return 'read'
            elif verb_stem == 'come':
                return 'came'
            else:
                return verb_stem + 'ed'
        elif tense == 'future':
            return 'will ' + verb_stem
        else:  # present
            return verb_stem + 's'  # Simple present 3rd person
    
    def translate_with_tense(self, korean_text):
        """
        Translate with proper tense handling
        """
        print(f"\nKorean: {korean_text}")
        
        # Detect tense
        tense = self.detect_tense(korean_text)
        print(f"Detected tense: {tense}")
        
        # Extract components
        components = self.extract_sentence_components(korean_text)
        
        # Translate
        translated = {
            'subject': [self.dictionary.get(w, w) for w in components['subject']],
            'verb': [self.dictionary.get(w, w) for w in components['verb']],
            'object': [self.dictionary.get(w, w) for w in components['object']],
            'time': [self.dictionary.get(w, w) for w in components['time']],
        }
        
        # Conjugate verb
        if translated['verb']:
            verb = translated['verb'][0]
            conjugated = self.conjugate_verb(verb, tense)
            translated['verb'] = [conjugated]
        
        # Build sentence
        english_parts = []
        
        if translated['subject']:
            english_parts.append(' '.join(translated['subject']))
        
        if translated['verb']:
            english_parts.append(' '.join(translated['verb']))
        
        if translated['object']:
            english_parts.append(' '.join(translated['object']))
        
        if translated['time']:
            english_parts.append(' '.join(translated['time']))
        
        english = ' '.join(english_parts) + '.'
        
        print(f"English: {english}")
        return english

# Test tense-aware translator
tense_translator = TenseAwareTranslator()

tense_examples = [
    "나는 밥을 먹는다.",      # present
    "나는 밥을 먹었다.",      # past
    "나는 밥을 먹겠다.",      # future
]

for sent in tense_examples:
    tense_translator.translate_with_tense(sent)
    print()

# ============================================================================
# 5. COMPLETE TRANSLATOR CLASS
# ============================================================================
print("\n5. COMPLETE TRANSLATOR (PUTTING IT ALL TOGETHER)")
print("-" * 100)

class KoreanEnglishTranslator:
    """
    A complete basic Korean-English translator
    """
    
    def __init__(self):
        self.mecab = Mecab()
        
        # Expanded dictionary
        self.dictionary = {
            # Pronouns
            '나': 'I', '저': 'I', '너': 'you', '우리': 'we',
            
            # Nouns
            '밥': 'rice', '음식': 'food', '책': 'book', '학생': 'student',
            '학교': 'school', '친구': 'friend', '영화': 'movie',
            '한국어': 'Korean', '선생님': 'teacher',
            
            # Verbs
            '먹': 'eat', '가': 'go', '보': 'watch', '읽': 'read',
            '공부': 'study', '하': 'do', '좋아하': 'like',
            
            # Adjectives
            '예쁘': 'pretty', '좋': 'good', '맛있': 'delicious',
            
            # Time
            '어제': 'yesterday', '오늘': 'today', '내일': 'tomorrow',
            
            # Prepositions
            '와': 'with', '과': 'with',
        }
        
        self.verb_conjugations = {
            ('eat', 'past'): 'ate',
            ('go', 'past'): 'went',
            ('watch', 'past'): 'watched',
            ('read', 'past'): 'read',
            ('study', 'past'): 'studied',
            ('do', 'past'): 'did',
        }
    
    def translate(self, korean_text):
        """
        Main translation method
        """
        print(f"\n{'='*80}")
        print(f"TRANSLATING: {korean_text}")
        print('='*80)
        
        # Step 1: Morphological analysis
        morphs = self.mecab.pos(korean_text)
        print(f"\nStep 1: Morphological Analysis")
        for word, pos in morphs:
            print(f"  {word:<10} {pos}")
        
        # Step 2: Extract components
        print(f"\nStep 2: Extract Sentence Components")
        subject, obj, verb, time, tense = self._extract_components(morphs)
        print(f"  Subject: {subject}")
        print(f"  Object: {obj}")
        print(f"  Verb: {verb}")
        print(f"  Time: {time}")
        print(f"  Tense: {tense}")
        
        # Step 3: Translate components
        print(f"\nStep 3: Translate Each Component")
        trans_subj = self._translate_list(subject)
        trans_obj = self._translate_list(obj)
        trans_verb = self._translate_verb(verb, tense)
        trans_time = self._translate_list(time)
        
        print(f"  Subject: {subject} → {trans_subj}")
        print(f"  Object: {obj} → {trans_obj}")
        print(f"  Verb: {verb} → {trans_verb}")
        print(f"  Time: {time} → {trans_time}")
        
        # Step 4: Build English sentence (SVO order)
        print(f"\nStep 4: Reorder to English (SVO)")
        english_parts = [p for p in [trans_subj, trans_verb, trans_obj, trans_time] if p]
        english = ' '.join(english_parts) + '.'
        
        print(f"\nFINAL TRANSLATION: {english}")
        print('='*80)
        
        return english
    
    def _extract_components(self, morphs):
        """Extract sentence components from morphemes"""
        subject, obj, verb, time = [], [], [], []
        tense = 'present'
        
        current_noun = []
        
        for i, (word, pos) in enumerate(morphs):
            # Collect nouns
            if pos.startswith('N'):
                current_noun.append(word)
            
            # Check particle to determine role
            elif pos in ['JKS', 'JX']:  # Subject marker
                subject = current_noun
                current_noun = []
            elif pos == 'JKO':  # Object marker
                obj = current_noun
                current_noun = []
            
            # Verb
            elif pos in ['VV', 'VA']:
                verb.append(word)
            
            # Tense marker
            elif pos == 'EP':
                if word in ['었', '았']:
                    tense = 'past'
            
            # Time adverbs
            elif pos == 'MAG' and word in ['어제', '오늘', '내일']:
                time.append(word)
        
        return subject, obj, verb, time, tense
    
    def _translate_list(self, words):
        """Translate a list of words"""
        if not words:
            return ''
        translated = [self.dictionary.get(w, w) for w in words]
        return ' '.join(translated)
    
    def _translate_verb(self, verb_list, tense):
        """Translate and conjugate verb"""
        if not verb_list:
            return ''
        
        verb_stem = verb_list[0]
        english_verb = self.dictionary.get(verb_stem, verb_stem)
        
        # Apply conjugation
        if (english_verb, tense) in self.verb_conjugations:
            return self.verb_conjugations[(english_verb, tense)]
        elif tense == 'past':
            return english_verb + 'ed'
        else:
            return english_verb + 's'

# Test complete translator
complete_translator = KoreanEnglishTranslator()

final_examples = [
    "나는 밥을 먹는다.",
    "학생이 책을 읽었다.",
    "친구와 영화를 봤어요.",
    "어제 학교에 갔다.",
]

for sent in final_examples:
    complete_translator.translate(sent)

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
   - Only handles basic SOV → SVO
   - Doesn't handle complex sentences, clauses
   
3. NO CONTEXT
   - Can't handle dropped subjects
   - Can't resolve ambiguity
   
4. BASIC TENSE
   - Only present/past
   - No progressive, perfect tenses
   
5. NO IDIOMS
   - Translates literally
   - Misses cultural expressions
   
6. SIMPLIFIED CONJUGATION
   - English verb conjugation is simplified
   - Doesn't handle irregular verbs well

Improvements needed:
✓ Use machine translation APIs (Google Translate, Papago)
✓ Add neural machine translation models
✓ Use larger dictionaries/databases
✓ Add context understanding
✓ Handle multiple sentence structures
✓ Include idiom translation
✓ Better verb conjugation system
""")

# ============================================================================
# 7. HOMEWORK
# ============================================================================
print("\n7. HOMEWORK")
print("-" * 100)
print("""
Complete these exercises to improve your translator:

1. EXPAND VOCABULARY
   Add 50 more words to the dictionary in different categories:
   - 10 more nouns
   - 10 more verbs
   - 10 more adjectives
   - 10 time/place words
   - 10 common phrases

2. HANDLE QUESTIONS
   Modify translator to detect and translate questions (ending in 니/까?)
   Example: "밥을 먹었니?" → "Did you eat rice?"

3. ADD PLURAL HANDLING
   Detect plural marker (들) and translate correctly
   Example: "친구들" → "friends"

4. IMPROVE VERB CONJUGATION
   Create a better verb conjugation system with irregular verbs

5. HANDLE COMPOUND SENTENCES
   Translate sentences with multiple clauses
   Example: "밥을 먹고 영화를 봤어요" → "I ate rice and watched a movie"

6. ADD ADJECTIVE HANDLING
   Properly translate adjective-noun phrases
   Example: "예쁜 꽃" → "pretty flower"

7. CHALLENGE: BIDIRECTIONAL TRANSLATOR
   Create an English → Korean translator
   Handle the reverse transformation (SVO → SOV)

8. INTEGRATION PROJECT
   Combine your translator with a GUI or web interface
   Allow users to input Korean text and see translation

9. EVALUATION SYSTEM
   Create test cases to measure translation accuracy
   Compare against Google Translate

10. RESEARCH PROJECT
    Research and write about:
    - How professional translation systems work
    - Neural machine translation
    - Attention mechanisms
    - Transformer models (like in Google Translate)
""")

print("\n" + "=" * 100)
print("DAY 5 COMPLETE!")
print("=" * 100)
print("""
🎉 CONGRATULATIONS! You've built a basic Korean-English translator!

You've learned:
✓ How to analyze Korean sentence structure
✓ How to extract sentence components (subject, object, verb)
✓ How to handle word order differences (SOV → SVO)
✓ How to detect and apply tense
✓ How to translate particles
✓ The challenges of real translation systems

Your 5-day journey:
Day 1: Tokenization ✓
Day 2: Named Entity Recognition ✓
Day 3: POS Tagging & Dependency Parsing ✓
Day 4: Korean NLP ✓
Day 5: Korean-English Translator ✓

Next steps:
- Continue expanding your translator
- Learn about neural machine translation
- Explore transformer models
- Build a web app for your translator
- Study advanced NLP topics

You now have the foundation to build real NLP applications!
Keep learning and building! 🚀
""")