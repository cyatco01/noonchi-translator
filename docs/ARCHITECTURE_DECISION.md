# Architecture Decision: Translation Engine Approach

## The Question
Should we use Papago API + korean-conjugator, or build traditional NLP translation from scratch?

## Answer: Both (Hybrid Approach)

### Strategic Reasoning

This project has **two goals**:
1. **Learning Goal**: Master NLP/computational linguistics concepts
2. **Portfolio Goal**: Build a working, impressive web application

A hybrid approach achieves both while demonstrating sophisticated thinking.

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NOONCHI TRANSLATOR                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      FastAPI Backend                    │
        │      Two Translation Engines            │
        └─────────────────────────────────────────┘
                 │                    │
                 ▼                    ▼
    ┌────────────────────┐  ┌──────────────────────┐
    │  Traditional NLP   │  │   Production API     │
    │  (Educational)     │  │   (Practical)        │
    └────────────────────┘  └──────────────────────┘
```

---

## Engine 1: Traditional NLP (Learning Showcase)

### Location: `backend/nlp/traditional/`

### Purpose
Demonstrate understanding of fundamental NLP/MT techniques

### What It Teaches You

#### Module: Rule-Based Machine Translation (RBMT)
```
English Input → Morphological Analysis → Syntactic Parsing →
Transfer Rules → Target Generation → Korean Output
```

**NLP Concepts Covered**:
1. **Tokenization**: English text → tokens
2. **POS Tagging**: Identify nouns, verbs, adjectives
3. **Dependency Parsing**: Understanding sentence structure
4. **Bilingual Dictionaries**: Word-to-word mappings
5. **Transfer Rules**: English SVO → Korean SOV reordering
6. **Morphological Generation**: Korean conjugation rules
7. **Agreement Rules**: Subject-object-verb agreement

### Implementation Steps

#### Step 1: English Analysis Pipeline
```python
# File: backend/nlp/traditional/english_analyzer.py

import spacy
from typing import List, Dict

class EnglishAnalyzer:
    """
    Analyze English text for translation
    
    NLP Techniques:
    - Tokenization
    - POS tagging
    - Dependency parsing
    - Named entity recognition
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def analyze(self, text: str) -> Dict:
        """
        Full linguistic analysis of English sentence
        
        Returns:
            {
                'tokens': [...],
                'pos_tags': [...],
                'dependencies': [...],
                'entities': [...]
            }
        """
        doc = self.nlp(text)
        
        return {
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'dependencies': [(token.text, token.dep_, token.head.text) 
                           for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents]
        }
```

#### Step 2: Bilingual Dictionary
```python
# File: backend/nlp/traditional/dictionary.py

class BilingualDictionary:
    """
    English-Korean word mappings
    
    NLP Concept: Lexical Transfer
    In real MT systems, this would be a large database
    For learning, we'll implement common words
    """
    
    def __init__(self):
        self.dictionary = {
            # Verbs
            'eat': {'base': '먹다', 'honorific': '드시다'},
            'go': {'base': '가다', 'honorific': '가시다'},
            'sleep': {'base': '자다', 'honorific': '주무시다'},
            'drink': {'base': '마시다', 'honorific': '드시다'},
            'want': {'stem': '싶'},
            
            # Nouns
            'food': '음식',
            'water': '물',
            'home': '집',
            
            # Adjectives
            'good': '좋다',
            'bad': '나쁘다',
            
            # Question words
            'do': '하다',
            'you': '당신',  # Context-dependent in Korean
        }
    
    def lookup(self, word: str, pos: str = None) -> Dict:
        """
        Look up word translation
        
        Args:
            word: English word (lemmatized)
            pos: Part of speech (optional, for disambiguation)
        
        Returns:
            Korean translation(s) with metadata
        """
        return self.dictionary.get(word.lower(), None)
```

#### Step 3: Reordering Rules (English SVO → Korean SOV)
```python
# File: backend/nlp/traditional/reorder.py

class SyntaxReorderer:
    """
    Transform English word order to Korean word order
    
    NLP Concept: Syntactic Transfer
    
    English: Subject - Verb - Object (SVO)
    "I eat food"
    
    Korean: Subject - Object - Verb (SOV)
    "나는 음식을 먹어요"
    (I - food - eat)
    """
    
    def reorder_svo_to_sov(self, dependencies: List) -> List:
        """
        Reorder sentence based on dependency parse
        
        Learning Point: Korean syntax fundamentally different from English
        This is why neural MT is so powerful - it learns these patterns
        """
        # Extract subject, object, verb from dependencies
        subject = None
        obj = None
        verb = None
        
        for token, dep, head in dependencies:
            if dep == 'nsubj':  # Nominal subject
                subject = token
            elif dep == 'dobj':  # Direct object
                obj = token
            elif dep == 'ROOT':  # Main verb
                verb = token
        
        # Reorder: SVO → SOV
        korean_order = []
        if subject:
            korean_order.append(subject)
        if obj:
            korean_order.append(obj)
        if verb:
            korean_order.append(verb)
        
        return korean_order
```

#### Step 4: Korean Morphological Generator
```python
# File: backend/nlp/traditional/korean_generator.py

from korean_conjugator import conjugate

class KoreanGenerator:
    """
    Generate properly conjugated Korean from verb stems
    
    NLP Concept: Morphological Generation
    The reverse of morphological analysis (parsing)
    """
    
    def conjugate_verb(self, verb_stem: str, formality: str, 
                      tense: str = 'present', 
                      sentence_type: str = 'statement') -> str:
        """
        Generate Korean verb form
        
        Args:
            verb_stem: Korean verb stem (e.g., '먹')
            formality: 'formal', 'polite', 'casual'
            tense: 'present', 'past', 'future'
            sentence_type: 'statement', 'question'
        
        Returns:
            Conjugated verb (e.g., '먹습니다', '먹어요', '먹어')
        """
        # Use korean-conjugator library for actual conjugation
        # This library implements Korean conjugation rules
        
        # Map our formality levels to conjugation types
        formality_map = {
            'formal': 'formal_high',
            'polite': 'informal_high', 
            'casual': 'informal_low'
        }
        
        result = conjugate(
            verb_stem + '다',  # Add dictionary form ending
            tense=tense,
            formality=formality_map[formality],
            sentence_type=sentence_type
        )
        
        return result
```

#### Step 5: Complete Traditional MT Pipeline
```python
# File: backend/nlp/traditional/translator.py

class TraditionalTranslator:
    """
    Complete rule-based machine translation pipeline
    
    Educational Implementation of Classical NLP/MT
    
    Pipeline:
    1. Analyze English (tokenize, parse, tag)
    2. Look up words in dictionary
    3. Apply transfer rules (reordering)
    4. Generate Korean morphology
    5. Apply formality rules
    """
    
    def __init__(self):
        self.analyzer = EnglishAnalyzer()
        self.dictionary = BilingualDictionary()
        self.reorderer = SyntaxReorderer()
        self.generator = KoreanGenerator()
    
    def translate(self, english_text: str, 
                  relationship: str = 'colleague') -> Dict:
        """
        Translate English to Korean using traditional NLP techniques
        
        Returns:
            {
                'translation': '...',
                'steps': {...},  # Show intermediate steps
                'method': 'traditional_nlp'
            }
        """
        # Step 1: Analyze English
        analysis = self.analyzer.analyze(english_text)
        
        # Step 2: Look up words
        translations = []
        for lemma, pos in zip(analysis['lemmas'], 
                             [p[1] for p in analysis['pos_tags']]):
            korean = self.dictionary.lookup(lemma, pos)
            translations.append(korean)
        
        # Step 3: Reorder syntax
        korean_order = self.reorderer.reorder_svo_to_sov(
            analysis['dependencies']
        )
        
        # Step 4: Generate Korean morphology
        formality = self._map_relationship_to_formality(relationship)
        
        # Step 5: Combine and conjugate
        final_translation = self._assemble_sentence(
            korean_order, 
            translations,
            formality
        )
        
        return {
            'translation': final_translation,
            'steps': {
                'analysis': analysis,
                'word_translations': translations,
                'reordered': korean_order,
                'formality': formality
            },
            'method': 'traditional_nlp',
            'note': 'This is a simplified educational implementation'
        }
```

### Why This is Valuable for Learning

1. **Shows you understand MT evolution**: From rule-based → statistical → neural
2. **Demonstrates linguistic knowledge**: Syntax, morphology, transfer
3. **Portfolio talking point**: "I built both approaches to understand trade-offs"
4. **Hands-on with NLP fundamentals**: Not just using APIs

### Limitations (Be Honest About These)

- ❌ Limited vocabulary (only common words)
- ❌ Simple syntax handling (complex sentences will fail)
- ❌ No idiom handling
- ❌ Won't match production quality

**But that's okay!** This is about demonstrating understanding, not competing with Google.

---

## Engine 2: Production API (Practical)

### Location: `backend/translation/production/`

### Purpose
Build a working, high-quality translation system for the web app

### Architecture

```
User Input (English + Relationship)
         ↓
   Papago API Translation
         ↓
   Korean Text (Default Polite Form)
         ↓
   KoNLPy Morphological Analysis
         ↓
   korean-conjugator for Formality Adjustment
         ↓
   Final Korean Output (Correct Formality)
```

### Implementation

```python
# File: backend/translation/production/translator.py

import requests
from konlpy.tag import Mecab
from korean_conjugator import conjugate

class ProductionTranslator:
    """
    Production-quality translation using Papago API + formality adjustment
    
    Why this approach:
    - Papago provides excellent base translation
    - We add unique value: context-aware formality
    - Best user experience
    """
    
    def __init__(self, papago_client_id: str, papago_client_secret: str):
        self.client_id = papago_client_id
        self.client_secret = papago_client_secret
        self.mecab = Mecab()
        self.base_url = "https://openapi.naver.com/v1/papago/n2mt"
    
    def translate(self, english_text: str, relationship: str) -> Dict:
        """
        Translate with context-aware formality
        
        Steps:
        1. Get base translation from Papago
        2. Parse Korean morphology
        3. Adjust formality based on relationship
        4. Return culturally appropriate result
        """
        # Step 1: Base translation
        base_korean = self._call_papago_api(english_text)
        
        # Step 2: Parse
        morphemes = self.mecab.pos(base_korean)
        
        # Step 3: Adjust formality
        formality = self._map_relationship(relationship)
        adjusted_korean = self._adjust_formality(base_korean, formality)
        
        return {
            'translation': adjusted_korean,
            'base_translation': base_korean,
            'formality_level': formality,
            'method': 'production_api'
        }
    
    def _call_papago_api(self, text: str) -> str:
        """Call Papago translation API"""
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        data = {
            "source": "en",
            "target": "ko",
            "text": text
        }
        response = requests.post(self.base_url, headers=headers, data=data)
        return response.json()['message']['result']['translatedText']
```

---

## User Interface: Side-by-Side Comparison

### Frontend Feature: "Compare Translation Methods"

```jsx
// Show both translations side-by-side
<div className="translation-comparison">
  <div className="traditional-result">
    <h3>Traditional NLP Method</h3>
    <p>{traditionalTranslation}</p>
    <button onClick={showSteps}>See Translation Steps</button>
  </div>
  
  <div className="production-result">
    <h3>Production Quality (Papago + Formality)</h3>
    <p>{productionTranslation}</p>
    <span className="badge">Recommended for actual use</span>
  </div>
</div>
```

This lets users (and portfolio reviewers) see the difference!

---

## Portfolio Presentation

### README Section: "Dual Translation Engines"

```markdown
## Translation Approaches

This project implements **two translation engines** to demonstrate both 
traditional NLP fundamentals and modern production practices:

### 1. Traditional NLP Engine (Educational)
Built from scratch using:
- spaCy for English linguistic analysis
- Custom bilingual dictionaries
- Syntax reordering rules (SVO → SOV)
- Morphological generation with korean-conjugator

**Purpose**: Demonstrates understanding of classical MT techniques and 
computational linguistics fundamentals.

**Strengths**: Shows deep NLP knowledge, transparent process  
**Limitations**: Limited vocabulary, simple syntax only

### 2. Production Engine (Practical)
Combines industry tools:
- Papago API for high-quality base translation
- KoNLPy for Korean morphological analysis
- Custom formality transformation layer

**Purpose**: Provides reliable, accurate translations for real use.

**Strengths**: Production-quality results, handles complex sentences  
**Limitations**: Relies on external API (black box)

### Why Both?
This architecture demonstrates:
- Understanding of when to build vs. when to integrate
- Knowledge of MT evolution (rule-based → neural)
- Practical engineering judgment
- Focus on unique value-add (formality adaptation)
```

---

## Recommended Learning Path

### Phase 1: Build Traditional Engine (2-3 weeks)
**Focus**: Learn NLP fundamentals deeply
- Implement each component with detailed comments
- Test with simple sentences
- Document what works and what doesn't

### Phase 2: Implement Production Engine (1 week)
**Focus**: Practical engineering
- Integrate Papago API
- Build formality adjustment layer
- Make it production-ready

### Phase 3: Comparison & Documentation (3-4 days)
**Focus**: Portfolio presentation
- Side-by-side UI
- Write detailed comparison in README
- Create blog post explaining approach

---

## Interview Talking Points

When discussing this project:

**"I built two translation engines to understand the trade-offs..."**

1. **Traditional approach taught me**: Tokenization, parsing, transfer rules, morphological generation

2. **Production approach taught me**: API integration, focusing on unique value-add, engineering judgment

3. **The comparison demonstrates**: Understanding of when to build vs. integrate, evolution of MT techniques, practical problem-solving

4. **The result**: A working app that uses the production engine, plus an educational showcase of NLP fundamentals

---

## My Recommendation

✅ **Build both engines**

Why:
1. **Learning depth**: Traditional engine teaches you NLP thoroughly
2. **Practical results**: Production engine makes your app actually useful
3. **Portfolio strength**: Shows range of skills and judgment
4. **Interview material**: Great talking points about trade-offs

Start with the traditional engine (deeper learning), then add production engine (practical results).

---

## Next Steps

Would you like me to:
1. Start with Module 1: Building the Traditional NLP Engine?
2. Start with Module 2: Setting up the Production Engine?
3. Create the project structure for both engines first?

My suggestion: Let's set up the dual-engine architecture, then build the traditional engine module-by-module to maximize learning!
