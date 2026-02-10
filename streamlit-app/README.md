# Neotericos Streamlit App

Simple, standalone clinical evidence search app with Streamlit.

## 🚀 Quick Setup

### 1. Copy ChromaDB Storage

```bash
# From project root
cp -r backend/chromadb_storage streamlit-app/
```

Or on Windows:
```bash
xcopy backend\chromadb_storage streamlit-app\chromadb_storage\ /E /I
```

### 2. Create .env File

```bash
cd streamlit-app
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_key_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## ✨ Features

- ✅ **All-in-One**: No separate backend needed
- ✅ **Chat Interface**: Clean, simple UI
- ✅ **Real-time Search**: Instant clinical evidence
- ✅ **Source Citations**: See where answers come from
- ✅ **Conversation Memory**: Maintains context
- ✅ **Easy Deployment**: One command to run

## 📁 Structure

```
streamlit-app/
├── app.py                  # Main Streamlit app
├── chromadb_storage/       # Vector database (copy from backend)
├── requirements.txt        # Dependencies
├── .env                    # API key (create this)
└── README.md              # This file
```

## 🌐 Deployment Options

### Streamlit Cloud (Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add `ANTHROPIC_API_KEY` in secrets
5. Deploy!

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t neotericos .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your_key neotericos
```

### Heroku

```bash
# Install Heroku CLI, then:
heroku create neotericos-app
heroku config:set ANTHROPIC_API_KEY=your_key
git push heroku main
```

## 💡 Usage

1. **Click "Initialize RAG System"** in sidebar
2. **Wait** for embeddings and database to load
3. **Type your question** in the chat input
4. **Get evidence-based answers** with sources!

## 🎯 Example Questions

- What are the treatment options for diabetes?
- Evidence for metformin in type 2 diabetes?
- Latest hypertension management guidelines?
- Compare rheumatoid arthritis treatments

## ⚙️ Configuration

Edit `app.py` to change:
- Model: `model="claude-sonnet-4-5"`
- Top-K results: `top_k=5`
- Max tokens: `max_tokens=2000`
- Temperature: `temperature=0.7`

## 🐛 Troubleshooting

### ChromaDB not found
- Make sure you copied `chromadb_storage/` folder
- Should be in same directory as `app.py`

### API Key error
- Check `.env` file exists
- Verify key is correct
- Try restarting the app

### Slow first query
- Embedding model loads on first use
- Subsequent queries are faster

## 📊 System Requirements

- Python 3.9+
- 2GB RAM minimum
- ~500MB disk space for embeddings

## 🔒 Security Notes

- Never commit `.env` file
- Use Streamlit secrets for deployment
- Keep API keys secure

## 🆚 vs FastAPI + Next.js

**Streamlit App:**
- ✅ Easier to deploy
- ✅ One command to run
- ✅ Built-in UI components
- ✅ Perfect for demos/MVPs

**FastAPI + Next.js:**
- ✅ More customizable
- ✅ Better for production
- ✅ Separate frontend/backend
- ✅ RESTful API
