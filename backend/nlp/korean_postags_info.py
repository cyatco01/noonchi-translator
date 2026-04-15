from konlpy.tag import Mecab

# Korean POS Tag Reference Guide
print("=" * 100)
print("KOREAN POS TAGS - COMPLETE REFERENCE")
print("=" * 100)

def show_korean_pos_tags():
    """
    Complete list of Korean POS tags with explanations and examples
    """
    
    korean_pos_tags = {
        "NOUNS (명사)": {
            "NNG": ("General Noun", "일반 명사", ["사람", "책", "컴퓨터", "학교"]),
            "NNP": ("Proper Noun", "고유 명사", ["서울", "한국", "삼성", "김철수"]),
            "NNB": ("Bound Noun", "의존 명사", ["것", "데", "수", "등"]),
        },
        
        "PRONOUNS (대명사)": {
            "NP": ("Pronoun", "대명사", ["나", "너", "우리", "저", "그"]),
        },
        
        "NUMERALS (수사)": {
            "NR": ("Numeral", "수사", ["하나", "둘", "첫째", "일", "이"]),
        },
        
        "VERBS (동사)": {
            "VV": ("Verb", "동사", ["먹다", "가다", "보다", "하다"]),
            "VA": ("Adjective", "형용사", ["예쁘다", "크다", "좋다", "작다"]),
            "VX": ("Auxiliary Verb/Adjective", "보조 용언", ["있다", "없다", "계시다"]),
            "VCP": ("Positive Copula", "긍정 지정사", ["이다"]),
            "VCN": ("Negative Copula", "부정 지정사", ["아니다"]),
        },
        
        "ADVERBS (부사)": {
            "MAG": ("General Adverb", "일반 부사", ["매우", "아주", "잘", "빨리", "천천히"]),
            "MAJ": ("Conjunctive Adverb", "접속 부사", ["그러나", "그리고", "또한", "하지만"]),
        },
        
        "INTERJECTIONS (감탄사)": {
            "IC": ("Interjection", "감탄사", ["아", "어", "오", "와"]),
        },
        
        "PARTICLES (조사)": {
            "JKS": ("Subject Particle", "주격 조사", ["이", "가"]),
            "JKC": ("Complement Particle", "보격 조사", ["이", "가"]),
            "JKG": ("Possessive Particle", "관형격 조사", ["의"]),
            "JKO": ("Object Particle", "목적격 조사", ["을", "를"]),
            "JKB": ("Adverbial Particle", "부사격 조사", ["에", "에서", "으로", "로"]),
            "JKV": ("Vocative Particle", "호격 조사", ["아", "야", "여"]),
            "JKQ": ("Quotative Particle", "인용격 조사", ["라고", "고"]),
            "JX": ("Auxiliary Particle", "보조사", ["는", "도", "만", "부터", "까지"]),
            "JC": ("Conjunction Particle", "접속 조사", ["와", "과", "하고", "이나", "나"]),
        },
        
        "VERB ENDINGS (어미)": {
            "EP": ("Pre-final Ending", "선어말 어미", ["시", "었", "겠"]),
            "EF": ("Final Ending", "종결 어미", ["다", "요", "니", "까"]),
            "EC": ("Connecting Ending", "연결 어미", ["고", "서", "면", "지만", "는데"]),
            "ETN": ("Nominal Ending", "명사형 전성 어미", ["ㄴ", "기", "음"]),
            "ETM": ("Adnominal Ending", "관형사형 전성 어미", ["ㄴ", "는", "ㄹ", "던"]),
        },
        
        "PREFIXES (접두사)": {
            "XPN": ("Noun Prefix", "체언 접두사", ["새", "헛", "맏"]),
        },
        
        "SUFFIXES (접미사)": {
            "XSN": ("Noun Suffix", "명사 파생 접미사", ["이", "님", "꾼", "쟁이"]),
            "XSV": ("Verb Suffix", "동사 파생 접미사", ["하다", "되다", "시키다"]),
            "XSA": ("Adjective Suffix", "형용사 파생 접미사", ["하다", "스럽다", "롭다"]),
        },
        
        "ROOTS (어근)": {
            "XR": ("Root", "어근", ["곱", "하", "푸르"]),
        },
        
        "DETERMINERS (관형사)": {
            "MM": ("Determiner", "관형사", ["이", "그", "저", "모든", "새"]),
        },
        
        "SYMBOLS & PUNCTUATION": {
            "SF": ("Final Punctuation", "마침표, 물음표, 느낌표", [".", "?", "!"]),
            "SP": ("Comma, Slash, Colon", "쉼표, 빗금, 콜론", [",", "/", ":", ";"]),
            "SS": ("Quote, Parenthesis", "따옴표, 괄호", ["\"", "'", "(", ")", "[", "]"]),
            "SE": ("Ellipsis", "줄임표", ["…", "..."]),
            "SO": ("Other Symbol", "기타 기호", ["@", "#", "&", "*"]),
            "SL": ("Foreign Language", "외국어", ["English", "words"]),
            "SH": ("Chinese Character", "한자", ["韓", "國"]),
            "SN": ("Number", "숫자", ["1", "2", "100", "2024"]),
        },
        
        "SPECIAL": {
            "NF": ("Unknown/Foreign", "분석 불능 외국어", ["unknown", "words"]),
            "NV": ("Unknown/Verb", "분석 불능 용언", []),
            "NA": ("Unknown", "분석 불능", []),
        }
    }
    
    for category, tags in korean_pos_tags.items():
        print(f"\n{'=' * 100}")
        print(f"{category}")
        print('=' * 100)
        print(f"{'Tag':<10} {'English':<30} {'Korean':<25} {'Examples'}")
        print('-' * 100)
        
        for tag, (eng_name, kor_name, examples) in tags.items():
            examples_str = ", ".join(examples[:3]) if examples else ""
            print(f"{tag:<10} {eng_name:<30} {kor_name:<25} {examples_str}")

def demonstrate_pos_tags():
    """
    Demonstrate POS tags with real Korean sentences
    """
    print("\n\n" + "=" * 100)
    print("DEMONSTRATION WITH REAL SENTENCES")
    print("=" * 100)
    
    mecab = Mecab()
    
    examples = [
        ("나는 학생입니다.", "I am a student."),
        ("어제 친구와 영화를 봤어요.", "Yesterday I watched a movie with a friend."),
        ("서울은 한국의 수도입니다.", "Seoul is the capital of Korea."),
        ("빨리 뛰어가세요!", "Run quickly!"),
        ("책을 읽는 것이 좋아요.", "I like reading books."),
        ("그 사람은 매우 친절해요.", "That person is very kind."),
    ]
    
    for korean, english in examples:
        print(f"\n{'-' * 100}")
        print(f"Korean:  {korean}")
        print(f"English: {english}")
        print(f"\n{'Morpheme':<20} {'POS':<10} {'Type'}")
        print('-' * 50)
        
        for word, pos in mecab.pos(korean):
            pos_type = get_pos_type(pos)
            print(f"{word:<20} {pos:<10} {pos_type}")

def get_pos_type(pos_tag):
    """Helper function to explain POS tag type"""
    pos_categories = {
        'NN': 'Noun',
        'NP': 'Pronoun',
        'NR': 'Numeral',
        'VV': 'Verb',
        'VA': 'Adjective',
        'VX': 'Auxiliary Verb',
        'VC': 'Copula',
        'MA': 'Adverb',
        'IC': 'Interjection',
        'JK': 'Particle',
        'JX': 'Auxiliary Particle',
        'JC': 'Conjunction Particle',
        'EP': 'Pre-final Ending',
        'EF': 'Final Ending',
        'EC': 'Connecting Ending',
        'ET': 'Nominal/Adnominal Ending',
        'XP': 'Prefix',
        'XS': 'Suffix',
        'XR': 'Root',
        'MM': 'Determiner',
        'SF': 'Final Punctuation',
        'SP': 'Comma/Punctuation',
        'SS': 'Quote/Bracket',
        'SE': 'Ellipsis',
        'SO': 'Other Symbol',
        'SL': 'Foreign Language',
        'SH': 'Chinese Character',
        'SN': 'Number',
        'NF': 'Unknown',
    }
    
    for prefix, category in pos_categories.items():
        if pos_tag.startswith(prefix):
            return category
    return 'Other'

def common_patterns():
    """
    Show common POS patterns in Korean
    """
    print("\n\n" + "=" * 100)
    print("COMMON POS PATTERNS")
    print("=" * 100)
    
    patterns = {
        "Subject Marking": {
            "pattern": "NNG/NNP + JKS",
            "example": "학생이 (student + subject marker)",
            "components": [("학생", "NNG"), ("이", "JKS")]
        },
        "Topic Marking": {
            "pattern": "NP + JX",
            "example": "나는 (I + topic marker)",
            "components": [("나", "NP"), ("는", "JX")]
        },
        "Object Marking": {
            "pattern": "NNG + JKO",
            "example": "책을 (book + object marker)",
            "components": [("책", "NNG"), ("을", "JKO")]
        },
        "Verb Conjugation": {
            "pattern": "VV + EP + EF",
            "example": "먹었다 (ate)",
            "components": [("먹", "VV"), ("었", "EP"), ("다", "EF")]
        },
        "Polite Ending": {
            "pattern": "VV + EP + EF",
            "example": "먹었어요 (ate - polite)",
            "components": [("먹", "VV"), ("었", "EP"), ("어요", "EF")]
        },
        "Adjective + Noun": {
            "pattern": "VA/MM + NNG",
            "example": "예쁜 꽃 (beautiful flower)",
            "components": [("예쁘", "VA"), ("ㄴ", "ETM"), ("꽃", "NNG")]
        },
        "Adverb + Verb": {
            "pattern": "MAG + VV",
            "example": "빨리 가다 (go quickly)",
            "components": [("빨리", "MAG"), ("가", "VV")]
        },
    }
    
    for pattern_name, details in patterns.items():
        print(f"\n{pattern_name}:")
        print(f"  Pattern: {details['pattern']}")
        print(f"  Example: {details['example']}")
        print(f"  Components: {details['components']}")

def particle_deep_dive():
    """
    Detailed look at Korean particles
    """
    print("\n\n" + "=" * 100)
    print("PARTICLE DEEP DIVE (조사)")
    print("=" * 100)
    
    particles = {
        "Subject Particles (주격 조사) - JKS": {
            "이/가": "marks the grammatical subject",
            "examples": ["학생이 공부한다 (The student studies)", "꽃이 예쁘다 (The flower is pretty)"]
        },
        "Object Particles (목적격 조사) - JKO": {
            "을/를": "marks the direct object",
            "examples": ["책을 읽다 (read a book)", "밥을 먹다 (eat rice)"]
        },
        "Topic Particles (보조사) - JX": {
            "은/는": "marks the topic (what the sentence is about)",
            "도": "also, too",
            "만": "only",
            "까지": "until, even",
            "부터": "from",
            "examples": ["나는 학생이다 (As for me, I'm a student)", "나도 간다 (I also go)"]
        },
        "Possessive Particle (관형격 조사) - JKG": {
            "의": "possessive marker (like 's in English)",
            "examples": ["나의 책 (my book)", "한국의 수도 (Korea's capital)"]
        },
        "Adverbial Particles (부사격 조사) - JKB": {
            "에": "at, to, in (location/time)",
            "에서": "at, in (location of action)",
            "으로/로": "to, toward, by means of",
            "examples": ["학교에 가다 (go to school)", "집에서 공부하다 (study at home)"]
        },
        "Conjunction Particles (접속 조사) - JC": {
            "와/과": "and, with",
            "하고": "and, with",
            "이나/나": "or",
            "examples": ["친구와 영화 (movie with friend)", "커피나 차 (coffee or tea)"]
        },
    }
    
    for particle_type, details in particles.items():
        print(f"\n{particle_type}")
        print("-" * 80)
        for particle, explanation in details.items():
            if particle != "examples":
                print(f"  {particle}: {explanation}")
        if "examples" in details:
            print(f"  Examples:")
            for ex in details["examples"]:
                print(f"    - {ex}")

if __name__ == "__main__":
    show_korean_pos_tags()
    demonstrate_pos_tags()
    common_patterns()
    particle_deep_dive()
    
    print("\n\n" + "=" * 100)
    print("KEY TAKEAWAYS")
    print("=" * 100)
    print("""
    Korean has ~45 POS tags vs English's 17 universal tags
    
    Most important tags for translation:
    ✓ NNG, NNP - Nouns
    ✓ VV, VA - Verbs and Adjectives
    ✓ JKS, JKO - Subject and Object particles
    ✓ JX - Topic/auxiliary particles
    ✓ EP, EF, EC - Verb endings (tense, politeness, connection)
    ✓ MAG - Adverbs
    
    Particles (조사) are unique to Korean:
    - Mark grammatical roles (subject, object, topic)
    - No direct English equivalent
    - Critical for understanding sentence structure
    - Must be handled carefully in translation
    """) 