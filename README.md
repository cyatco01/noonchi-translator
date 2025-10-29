# Noonchi Translator

An intelligent English-to-Korean translator that applies appropriate Korean speech levels (formality/honorifics) based on social relationships.

## About

Korean has seven distinct speech levels that express social hierarchy and politeness. Choosing the wrong level can be awkward or even offensive. **Noonchi** (눈치) means "social awareness" in Korean - the ability to read situations and relationships.

This project translates English to Korean while automatically applying the culturally appropriate level of formality based on who you're talking to.

### Example

**English Input**: "Do you want to eat?"

**To your boss** (하십시오체 - Formal):
```
드시고 싶으십니까?
```

**To a colleague** (해요체 - Polite):
```
먹고 싶어요?
```

**To a friend** (해체 - Casual):
```
먹고 싶어?
```

## Learning Goals

This is an educational project designed to explore:
- **Natural Language Processing (NLP)** fundamentals
- Korean linguistic concepts (morphology, honorifics, speech levels)
- Translation API integration (Papago/Google Translate)
- Full-stack web development (Python + React)
- Industry-standard software architecture

## How It Works

```
User Input → Relationship Selection → Translation API →
NLP Analysis → Formality Transformation → Korean Output
```

1. **User provides**: English text + relationship context (boss, colleague, friend, etc.)
2. **Base translation**: Papago API translates English → Korean (default polite form)
3. **Morphological analysis**: KoNLPy parses Korean text to identify verb stems and endings
4. **Formality transformation**: Custom rules adjust conjugations and vocabulary for target formality level
5. **Output**: Culturally appropriate Korean translation

## Tech Stack

### Backend
- **Python 3.9+** with FastAPI
- **KoNLPy** - Korean morphological analysis
- **Papago API** - Base translation service
- **Pydantic** - Data validation

### Frontend
- **React** - UI framework
- **Axios** - API communication
- **Tailwind CSS** - Styling

### NLP Concepts Applied
- Tokenization & morphological analysis
- Part-of-speech (POS) tagging
- Rule-based text transformation
- Entity-attribute mapping

## Project Status

🚧 **Currently in development** - MVP Phase

### MVP Features (In Progress)
- [x] Project planning and architecture
- [ ] Backend API setup with FastAPI
- [ ] Papago API integration
- [ ] Basic formality transformation (3 levels: formal, polite, casual)
- [ ] React frontend with relationship selector
- [ ] Verb conjugation rules for common verbs

### Future Enhancements
- [ ] All 7 formality levels
- [ ] Comprehensive verb conjugation database
- [ ] Honorific vocabulary mapping (먹다 → 드시다, 자다 → 주무시다)
- [ ] Sentence-level context analysis
- [ ] Multiple translation options display
- [ ] Conversation history mode
- [ ] ML-based formality detection from English text

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- Papago API key (get from [Naver Developers](https://developers.naver.com/products/papago/))

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd noonchi_project
```

2. **Backend setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend setup**
```bash
cd frontend
npm install
```

4. **Environment configuration**
```bash
cp .env.example .env
# Edit .env and add your Papago API credentials:
# PAPAGO_CLIENT_ID=your_client_id
# PAPAGO_CLIENT_SECRET=your_client_secret
```

### Running the Application

**Start the backend** (Terminal 1):
```bash
cd backend
source venv/bin/activate
uvicorn app:app --reload --port 8000
```

**Start the frontend** (Terminal 2):
```bash
cd frontend
npm start
```

Visit `http://localhost:3000` in your browser.

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm test
```

## Korean Speech Levels Reference

| Level | Name | Usage | Example Ending |
|-------|------|-------|----------------|
| 1 | 하십시오체 (hasipsioche) | Most formal - customers, elders | -습니다/-ㅂ니다 |
| 2 | 하오체 (haoche) | Formal - archaic, rarely used | -오/-소 |
| 3 | 하게체 (hageche) | Semi-formal - archaic | -네/-게 |
| 4 | 해라체 (haerache) | Plain - writing, children | -다/-라 |
| 5 | 해요체 (haeyoche) | Polite informal - most common | -아요/-어요 |
| 6 | 해체 (haeche) | Casual - friends, family | -아/-어 |
| 7 | Honorific | Special vocabulary & particles | 드시다, 주무시다 |

## Project Structure

```
noonchi_project/
├── backend/              # Python FastAPI application
│   ├── app.py           # Main API server
│   ├── translation/     # Translation & formality logic
│   ├── nlp/            # Korean NLP utilities (KoNLPy)
│   └── models/         # Data schemas
├── frontend/            # React web application
│   └── src/
│       ├── components/ # UI components
│       └── api.js     # Backend client
├── data/               # Language rules & mappings
│   ├── verb_conjugations.json
│   ├── honorific_vocab.json
│   └── relationship_map.json
├── tests/              # Test files
├── docs/               # Additional documentation
├── CLAUDE.md          # AI assistant development guide
└── README.md          # This file
```

## Contributing

This is a learning project, but suggestions and improvements are welcome!

If you'd like to contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Learning Resources

### Korean Language
- [Talk To Me In Korean](https://talktomeinkorean.com) - Comprehensive Korean learning
- [How to Study Korean](https://www.howtostudykorean.com) - Grammar and honorifics guide

### NLP & Korean Processing
- [KoNLPy Documentation](https://konlpy.org/en/latest/) - Korean NLP library
- [Korean Language Processing in Python](https://github.com/konlpy/konlpy)

### APIs & Frameworks
- [Papago API Docs](https://developers.naver.com/products/papago/) - Translation API
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Python web framework

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built as a learning project to explore NLP and Korean linguistics
- Inspired by the complexity and beauty of Korean honorific systems
- Thanks to the KoNLPy and FastAPI communities

---

**Note**: This project is under active development. The translation accuracy and formality transformations will improve as more linguistic rules are implemented.
