# PDF Farmer

PDF Farmer is a web application that helps you find and download PDF documents from across the internet based on your search keywords. It features an intelligent search algorithm that ranks results by relevance and provides a modern, Discord-inspired user interface.
###### Current Version: 0.5 (Early Development Stage)

## Features

- 🔍 Smart PDF search across multiple search engines
- 📊 Relevance scoring and ranking of results
- ⚡ Real-time streaming search results
- 📥 Bulk download capability for found PDFs
- 🧠 Intelligent keyword improvement suggestions
- 🎨 Modern Discord-inspired UI design
- 🔄 Rate limiting and request management
- ✨ Clean and responsive interface

## Tech Stack

- **Backend**: Python/Flask
- **Frontend**: HTML/CSS/JavaScript
- **Key Libraries**:
  - Natural Language Processing: NLTK, spaCy
  - Text Analysis: scikit-learn
  - Web Scraping: BeautifulSoup4, requests
  - Async Operations: aiohttp, asyncio

## Setup

1. Install Python dependencies:
```bash
pip install flask beautifulsoup4 requests scikit-learn nltk spacy aiohttp
python -m spacy download en_core_web_sm
```

2. Run the server:
```bash
python server.py
```

3. Open `index.html` in your web browser or serve it using a web server.

## Usage

1. Enter your search keyword in the search box
2. Adjust the search parameters if needed:
   - Maximum pages to search
   - Minimum relevance score
   - Show/hide low relevance results
3. Click "Search" to start finding PDFs
4. Use the "Download All" button to get multiple PDFs at once

## Project Structure

- `server.py`: Backend Flask server with PDF search and download logic
- `index.html`: Frontend interface with Discord-inspired design

## Features in Detail

### Smart Search
- Multi-engine PDF search
- Intelligent rate limiting
- Duplicate URL detection
- PDF verification

### Relevance Scoring
- TF-IDF vectorization
- Cosine similarity calculation
- Customizable relevance thresholds

### Keyword Enhancement
- Synonym expansion
- Context analysis
- Search term optimization

## Known Issues

### Performance
- Bulk PDF download process is currently slower than optimal due to sequential processing
- Large PDF files may cause temporary UI freezes during download
- Search results may take longer to appear with very broad search terms

### Browser Limitations
- Some browsers may limit the number of concurrent downloads
- Memory usage can spike when downloading multiple large PDFs

## Planned Features

### Performance Improvements
- ⚡ Parallel PDF downloading with progress indicators
- 🚀 Improved caching system for faster search results
- 📊 Download queue management system
- 🔄 Batch processing optimization

### New Features
- 📱 Mobile-responsive design improvements
- 🔍 Advanced search filters (by date, size, search engine)
- 📂 Custom download directory selection
- 💾 Save search results for later
- 📈 Search history and analytics

### UI Enhancements
- 🎨 Dark/Light theme toggle
- 📱 Better mobile experience
- 🔔 Download notifications
- 📊 Download progress visualization

## Support & Community

### GitHub Issues
- 🐛 Report bugs and issues through our [GitHub Issues](https://github.com/PDF-Farmer/issues)
- 💡 Submit feature requests and suggestions
- 📝 View existing issues and their status

### Discussion Forums
- 💬 Join our [GitHub Discussions](https://github.com/PDF-Farmer/discussions) for:
  - Q&A and troubleshooting
  - Feature requests and ideas
  - Share your use cases
  - Connect with other users

### Documentation
- 📚 Check our [Wiki](https://github.com/PDF-Farmer/wiki) for:
  - Detailed setup guides
  - Troubleshooting tips
  - Best practices
  - API documentation

### Stay Updated
- ⭐ Star the repository to receive notifications about major updates
- 👀 Watch the repository for detailed activity updates
- 🔔 Follow our [Release Notes](https://github.com/PDF-Farmer/releases) for version updates

### Contributing
- 🤝 We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md)
- 🌟 Check out our [Good First Issues](https://github.com/PDF-Farmer/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started

## License

This project is open source and available under the MIT License. 
