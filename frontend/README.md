# Noonchi Translator - Frontend

React-based frontend for the Noonchi Translator. This UI works with **all** translation backend methods through a unified API interface.

## Features

✨ **Multi-Backend Support** - Switch between different translation methods:
- AI Agent (Claude) - Currently available
- Papago + Rules - Coming soon
- Pure ML Model - Coming soon

✨ **Two-Step Workflow**
1. Set context (who you're talking to)
2. Translate with appropriate formality

✨ **Beautiful UI**
- Color-coded formality levels
- Real-time translation
- Translation history
- Copy to clipboard
- Explanations and pronunciation

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm build
```

The app will run at `http://localhost:3000`

**Make sure the backend is running!**
- Agent backend: `http://localhost:8000`
- Papago backend: `http://localhost:8001` (when implemented)
- ML backend: `http://localhost:8002` (when implemented)

## How to Use

### 1. Select Translation Method
Choose which backend to use from the dropdown at the top.

### 2. Set Context
- Select the relationship (boss, colleague, friend, etc.)
- Optionally describe the situation
- Click "Set Context"

### 3. Translate
- Enter English text
- Click "Translate"
- See Korean translation with:
  - Appropriate formality level
  - Romanization (pronunciation)
  - Explanation of linguistic choices

### 4. Keep Translating
- Same context persists for multiple translations
- Click "Change Context" to start fresh

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── FormalityBadge.jsx    # Color-coded formality indicator
│   ├── config/
│   │   └── backends.js            # Backend URLs & availability
│   ├── services/
│   │   └── api.js                 # API client (method-agnostic)
│   ├── styles/
│   │   └── index.css              # Tailwind CSS
│   ├── App.jsx                    # Main application
│   └── main.jsx                   # Entry point
├── public/
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

## Adding a New Translation Method

When you implement a new backend (e.g., Papago + Rules):

1. **Implement the backend** with these endpoints:
   - `POST /api/set-context` - Returns `{session_id, formality_level, ...}`
   - `POST /api/translate` - Returns `{translated_text, explanation, ...}`

2. **Update `src/config/backends.js`**:
   ```javascript
   papago_rules: {
     ...
     available: true,  // Change from false to true
     ...
   }
   ```

3. **That's it!** The frontend automatically supports it.

## API Contract

All backends must implement:

### POST /api/set-context
**Request:**
```json
{
  "relationship": "boss",
  "situation": "business meeting"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "relationship": "boss",
  "formality_level": "hasipsioche",
  "formality_description": "Using 하십시오체...",
  "situation": "business meeting",
  "message": "Context set!"
}
```

### POST /api/translate
**Request:**
```json
{
  "session_id": "uuid",
  "text": "Do you want to eat?"
}
```

**Response:**
```json
{
  "original_text": "Do you want to eat?",
  "translated_text": "드시고 싶으십니까?",
  "relationship": "boss",
  "formality_level": "hasipsioche",
  "explanation": "Used honorific verb...",
  "romanization": "deusigo sipsseumnikka?"
}
```

## Formality Levels

The app supports 3 primary Korean speech levels:

| Level | Korean | Color | Usage |
|-------|--------|-------|-------|
| hasipsioche | 하십시오체 | Blue | Boss, elder, customer, teacher |
| haeyoche | 해요체 | Green | Colleague, acquaintance |
| haeche | 해체 | Orange | Friend, younger sibling, child |

## Technologies

- **React 18** - UI framework
- **Vite** - Build tool (faster than Create React App)
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

## Development

### Hot Reload
Vite provides instant hot module replacement. Changes appear immediately!

### Environment Variables
Create `.env.local` if you need custom backend URLs:
```
VITE_AGENT_API_URL=http://localhost:8000
VITE_PAPAGO_API_URL=http://localhost:8001
VITE_ML_API_URL=http://localhost:8002
```

### Building
```bash
npm run build
```

Output goes to `dist/` directory.

## Troubleshooting

**"Network Error" / Can't connect to backend**
- Make sure backend server is running
- Check backend URL in `src/config/backends.js`
- Check browser console for CORS errors

**Session expired error**
- Sessions timeout after 30 minutes
- Click "Change Context" to create new session

**Translations not showing**
- Check that you set context first
- Verify backend is returning proper response format

## Future Enhancements

- [ ] Comparison mode (show all 3 methods side-by-side)
- [ ] Export translation history
- [ ] Dark mode
- [ ] Mobile app version
- [ ] Voice input
- [ ] Save favorite translations
- [ ] User authentication & history

---

**Made with ❤️  for learning Korean formality!**
