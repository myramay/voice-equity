# VoiceEquity 🗳️

> **Your civic document assistant — snap a photo, ask a question, get a plain answer in your language.**

VoiceEquity makes critical government forms, legal notices, and public signs accessible
to anyone — regardless of literacy level or English proficiency — using multimodal AI
and browser-native voice I/O.

---

## Example Use Cases

- **Eviction notice:** Tenant photographs a notice slid under the door, asks "What does this say I have to do?" in Spanish — VoiceEquity explains the deadline and reads the answer aloud in Spanish.
- **Benefits form:** A Somali refugee photographs a Medicaid application, asks "What information do they want here?" — receives a plain-language walkthrough with a clear next step.
- **Street sign / building notice:** A visually-complex posted notice is photographed; the user asks "Is this place open today?" in Haitian Creole and hears the answer spoken back.

---

## Quick Start

### 1. Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API key** → copy the key

> The free tier supports Gemma 3 models with generous rate limits for prototyping.
> Gemma 4 access uses the same key — switch `MODEL_NAME` in `main.py` once the model
> is publicly available (`gemma-4-it`).

### 2. Install Dependencies

```bash
cd voiceequity
pip install -r requirements.txt
```

### 3. Set the API Key

```bash
export GEMINI_API_KEY="your_key_here"
```

On Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY = "your_key_here"
```

### 4. Run the Server

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the App

Navigate to [http://localhost:8000](http://localhost:8000) in any modern browser.

---

## How It Works

```
Browser                       FastAPI                  Google GenAI
  │                              │                          │
  │── Upload photo (base64) ──►  │                          │
  │── Speak / type question ──►  │                          │
  │                              │── Gemma 4 multimodal ──► │
  │                              │◄── Plain-language ans ── │
  │◄── JSON answer ─────────────│                          │
  │── speechSynthesis reads aloud│
```

- **Voice input**: Web Speech API (built into Chrome / Edge / Safari) transcribes speech
  to text with no extra server round-trip.
- **Image**: Encoded as base64 and sent inline to the model along with the question.
- **Language detection**: Gemma auto-detects the language of the question and responds
  in the same language. An optional dropdown lets users hard-set the language for TTS
  voice selection.
- **TTS**: `window.speechSynthesis` reads the answer aloud automatically using the best
  matching installed voice for the detected language.

---

## Project Structure

```
voiceequity/
├── main.py          # FastAPI backend — /analyze + /health endpoints
├── index.html       # Single-page frontend (vanilla JS, no frameworks)
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable          | Required | Description                          |
|-------------------|----------|--------------------------------------|
| `GEMINI_API_KEY`  | Yes      | Your Google AI Studio API key        |

---

## Switching Gemma 4 Model Size

The default is `gemma-4-31b-it` (best accuracy). For faster / lower-cost responses,
change one line in `main.py`:

```python
MODEL_NAME = "gemma-4-e4b-it"   # efficient 4B variant, faster + cheaper
```

---

## Accessibility Notes

- Minimum 18 px font throughout
- All interactive elements are keyboard accessible
- `aria-live` regions announce dynamic content to screen readers
- Touch-friendly button sizes (min 54 px height)
- Mobile-first responsive layout works on any phone browser

---

## License

MIT
