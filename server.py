from flask import Flask, request, jsonify, Response, send_file
from bs4 import BeautifulSoup
import requests
import urllib.parse
import time
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import nltk
from nltk.corpus import wordnet
import itertools
import json
import spacy
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import zipfile
import io
from urllib.parse import unquote
from queue import Queue

app = Flask(__name__)

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize spaCy - with fallback
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy English language model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load('en_core_web_sm')

def is_valid_pdf_url(url):
    """Check if URL points to a PDF file and is accessible"""
    try:
        # Check file extension
        if not url.lower().endswith('.pdf'):
            return False
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf,*/*'
        }
        
        # Try HEAD request with shorter timeout
        try:
            response = requests.head(url, headers=headers, allow_redirects=True, timeout=3)
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/pdf' in content_type or 'pdf' in content_type:
                return True
                
        except requests.exceptions.RequestException:
            # If HEAD fails, we'll still consider it valid if it ends with .pdf
            return True
            
    except Exception as e:
        print(f"Error validating PDF URL {url}: {str(e)}")
        return False
    
    return False

def clean_text(text):
    """Clean and normalize text for ML processing"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def calculate_relevance_scores(keyword, pdf_items, min_relevance=0):
    """Calculate relevance scores for PDF results using TF-IDF"""
    if not pdf_items:
        return []
    
    # Clean the keyword and create document corpus
    cleaned_keyword = clean_text(keyword)
    
    # Prepare documents from titles and URLs
    documents = []
    for item in pdf_items:
        title = clean_text(item['title'])
        url_text = clean_text(item['url'].split('/')[-1].replace('.pdf', ''))
        # Combine title and URL text, giving more weight to title
        doc = f"{title} {title} {url_text}"  # Double weight for title
        documents.append(doc)
    
    # Add the keyword to the corpus
    all_docs = [cleaned_keyword] + documents
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(
        min_df=1,
        stop_words='english',
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Calculate cosine similarity between keyword and each document
    keyword_vector = tfidf_matrix[0:1]
    similarities = cosine_similarity(keyword_vector, tfidf_matrix[1:]).flatten()
    
    # Add scores to PDF items and sort by relevance
    scored_items = []
    for item, score in zip(pdf_items, similarities):
        item['relevance_score'] = float(score)
        scored_items.append(item)
    
    # Sort by relevance score
    scored_items.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Return all items without filtering
    return scored_items

def get_search_engines():
    """Get list of search engines with rate limit settings"""
    return [
        {
            'url': 'https://www.bing.com/search?q={}&count=50&first={}',
            'domain': 'bing.com',
            'selector': 'b_algo',
            'rate_limit': {'requests': 30, 'window': 60}  # 30 requests per minute
        },
        {
            'url': 'https://duckduckgo.com/html/?q={}+filetype:pdf&s={}',
            'domain': 'duckduckgo.com',
            'selector': 'result__body',
            'rate_limit': {'requests': 40, 'window': 60}  # 40 requests per minute
        },
        {
            'url': 'https://search.brave.com/search?q={}&offset={}',
            'domain': 'brave.com',
            'selector': 'result',
            'rate_limit': {'requests': 50, 'window': 60}  # 50 requests per minute
        }
    ]

class RateLimitHandler:
    def __init__(self):
        self.request_times = {}
        self.search_engines = get_search_engines()
        self.rate_limits = {
            engine['domain']: engine['rate_limit'] 
            for engine in self.search_engines
        }
        self.rate_limits['default'] = {'requests': 20, 'window': 60}
        
    async def get_available_engine(self):
        """Get the next available search engine"""
        now = time.time()
        
        for engine in self.search_engines:
            domain = engine['domain']
            if domain not in self.request_times:
                self.request_times[domain] = []
            
            # Clean old requests
            self.request_times[domain] = [
                t for t in self.request_times[domain] 
                if now - t < self.rate_limits[domain]['window']
            ]
            
            # Check if engine is available
            if len(self.request_times[domain]) < self.rate_limits[domain]['requests']:
                self.request_times[domain].append(now)
                return engine
                
        # If no engine is immediately available, find the one that will be available soonest
        wait_times = []
        for engine in self.search_engines:
            domain = engine['domain']
            if self.request_times[domain]:
                wait_time = self.request_times[domain][0] + self.rate_limits[domain]['window'] - now
                wait_times.append((wait_time, engine))
        
        if wait_times:
            wait_time, engine = min(wait_times, key=lambda x: x[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
            return engine
            
        return self.search_engines[0]  # Fallback to first engine

async def process_search_engine(engine_url, result_class, page, keyword, headers, seen_urls):
    """Process a single search engine page"""
    offset = page * 10
    search_url = engine_url.format(urllib.parse.quote(keyword + " filetype:pdf"), offset)
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(search_url, headers=headers, timeout=10) as response:
                if response.status == 429:  # Too Many Requests
                    print(f"Rate limit hit for {engine_url}")
                    return []
                
                html = await response.text()
                return await process_html(html, result_class, seen_urls)
        except Exception as e:
            print(f"Error fetching {search_url}: {str(e)}")
            return []

async def process_html(html, result_class, seen_urls):
    """Process HTML content for PDF links"""
    if not html:
        return []
        
    soup = BeautifulSoup(html, 'html.parser')
    results = soup.find_all(['li', 'div'], class_=result_class)
    
    pdf_results = []
    for result in results:
        try:
            link = result.find('a')
            if not link:
                continue
            
            url = link.get('href', '')
            if not url or url.lower() in seen_urls:
                continue
            
            url = urllib.parse.unquote(url)
            if url.lower().endswith('.pdf'):
                seen_urls.add(url.lower())
                pdf_results.append({
                    'url': url,
                    'title': link.get_text().strip() or url.split('/')[-1].replace('.pdf', '').replace('-', ' ')
                })
        except Exception as e:
            print(f"Error processing result: {str(e)}")
            continue
            
    return pdf_results

def fetch_pdfs(keyword, max_pages=1):
    """Search for PDFs with optimized speed and rate limiting"""
    pdf_links = []
    seen_urls = set()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }
    
    # Initialize rate limit handler
    rate_limit_handler = RateLimitHandler()
    
    # Run search with batching and rate limiting
    async def run_search():
        return await batch_search(keyword, max_pages, rate_limit_handler, headers, seen_urls)
    
    pdf_links = asyncio.run(run_search())
    
    # Apply ML-based filtering
    filtered_pdfs = calculate_relevance_scores(keyword, pdf_links)
    return filtered_pdfs

async def batch_search(keyword, max_pages, rate_limit_handler, headers, seen_urls):
    """Execute search in batches with smart engine selection"""
    BATCH_SIZE = 2  # Reduced batch size
    all_results = []
    
    for batch_start in range(0, max_pages, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, max_pages)
        print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1} (pages {batch_start+1}-{batch_end})...")
        
        tasks = []
        for page in range(batch_start, batch_end):
            # Get available search engine
            engine = await rate_limit_handler.get_available_engine()
            tasks.append(process_search_engine(
                engine['url'], 
                engine['selector'], 
                page, 
                keyword, 
                headers, 
                seen_urls
            ))
        
        # Execute batch
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and add valid results
        valid_results = [r for r in batch_results if isinstance(r, list)]
        all_results.extend([item for sublist in valid_results for item in sublist])
        
        # Add delay between batches
        if batch_end < max_pages:
            await asyncio.sleep(2)
    
    return all_results

def verify_pdf_url(url, headers):
    """Simple verification that URL ends with .pdf"""
    return url.lower().endswith('.pdf')

def add_pdf_to_results(url, link_element, pdf_links):
    """Add a PDF to the results list with proper title extraction"""
    # Skip if URL already exists
    if any(pdf['url'] == url for pdf in pdf_links):
        return
        
    # Get title from link text or URL
    title = link_element.get_text().strip()
    if not title or len(title) < 3:
        title = url.split('/')[-1].replace('.pdf', '').replace('-', ' ').replace('_', ' ')
    
    pdf_links.append({
        'url': url,
        'title': title
    })
    print(f"Added PDF: {title}")

def improve_keyword(keyword):
    """AI-powered keyword enhancement and expansion"""
    # Base keyword cleaning
    cleaned_keyword = clean_text(keyword)
    improved_keywords = [keyword]  # Always keep original
    
    try:
        # Process with spaCy
        doc = nlp(cleaned_keyword)
        
        # 1. Extract key phrases and entities
        key_phrases = []
        for chunk in doc.noun_chunks:
            key_phrases.append(chunk.text)
        for ent in doc.ents:
            key_phrases.append(ent.text)
            
        # 2. Domain-specific expansions
        domain_keywords = {
            'money': ['finance', 'investment', 'earnings', 'revenue', 'profit', 'income', 'wealth'],
            'business': ['enterprise', 'company', 'startup', 'entrepreneurship', 'management'],
            'learn': ['guide', 'tutorial', 'course', 'training', 'education', 'instruction'],
            'technology': ['software', 'digital', 'tech', 'system', 'application', 'development'],
            'research': ['study', 'analysis', 'investigation', 'report', 'findings', 'methodology'],
            'guide': ['manual', 'handbook', 'instructions', 'documentation', 'walkthrough']
        }
        
        # 3. Generate context-aware variations
        variations = []
        
        # Add domain-specific keywords
        for token in doc:
            if token.text.lower() in domain_keywords:
                variations.extend(domain_keywords[token.text.lower()])
        
        # Add noun phrase variations
        for phrase in key_phrases:
            variations.extend([
                f"{phrase} guide",
                f"{phrase} tutorial",
                f"how to {phrase}",
                f"{phrase} handbook",
                f"learn {phrase}",
                f"{phrase} for beginners",
                f"advanced {phrase}",
                f"{phrase} methodology"
            ])
        
        # 4. Add WordNet synonyms and related terms
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                synsets = wordnet.synsets(token.text)
                for syn in synsets[:3]:  # Limit to top 3 synsets
                    variations.extend(syn.lemma_names())
                    
                    # Add hypernyms (broader terms)
                    for hypernym in syn.hypernyms()[:2]:
                        variations.extend(hypernym.lemma_names())
        
        # 5. Add common search patterns
        patterns = [
            "guide to {}",
            "how to {}",
            "{} tutorial",
            "{} handbook",
            "{} methodology",
            "{} best practices",
            "understanding {}",
            "learn {}",
            "{} for beginners",
            "advanced {}"
        ]
        
        for pattern in patterns:
            variations.append(pattern.format(cleaned_keyword))
        
        # 6. Clean and normalize all variations
        variations = [clean_text(v) for v in variations]
        variations = [v for v in variations if v]  # Remove empty strings
        variations = list(set(variations))  # Remove duplicates
        
        # 7. Score variations using TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        try:
            tfidf_matrix = vectorizer.fit_transform([cleaned_keyword] + variations)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Combine with scores and sort
            scored_variations = list(zip(variations, similarities))
            scored_variations.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by minimum similarity but ensure variety
            min_similarity = 0.15  # Lower threshold for more variety
            relevant_variations = [v[0] for v in scored_variations if v[1] >= min_similarity]
            
            # Ensure minimum number of variations
            if len(relevant_variations) < 5:
                relevant_variations = [v[0] for v in scored_variations[:5]]
            
            # Always include original keyword
            if keyword not in relevant_variations:
                relevant_variations.insert(0, keyword)
            
            # Return top variations (increased from 8 to 12 for more coverage)
            return relevant_variations[:12]
            
        except Exception as e:
            print(f"Error in variation scoring: {str(e)}")
            return [keyword] + variations[:8]
            
    except Exception as e:
        print(f"Error in keyword enhancement: {str(e)}")
        return [keyword]  # Fallback to original keyword

def stream_search_results(keyword, max_pages, min_relevance=0, show_low_relevance=True):
    """Generator function to stream search results"""
    seen_urls = set()
    used_keywords = []
    
    # Get improved keywords
    improved_keywords = improve_keyword(keyword)
    print(f"\nImproved keywords: {improved_keywords}")
    
    yield json.dumps({
        'type': 'info',
        'message': f'Starting search with improved keywords: {", ".join(improved_keywords)}'
    }) + '\n'
    
    # Try each improved keyword
    for search_keyword in improved_keywords:
        print(f"\nTrying keyword: {search_keyword}")
        used_keywords.append(search_keyword)
        
        yield json.dumps({
            'type': 'info',
            'message': f'Searching with keyword: {search_keyword}'
        }) + '\n'
        
        pdf_links = fetch_pdfs(search_keyword, max_pages=max_pages)
        
        if pdf_links:
            # Score PDFs and stream them immediately
            scored_pdfs = calculate_relevance_scores(keyword, pdf_links, min_relevance=0)
            
            # Stream each PDF as it's found
            for pdf in scored_pdfs:
                if pdf['url'] not in seen_urls:
                    seen_urls.add(pdf['url'])
                    yield json.dumps({
                        'type': 'result',
                        'pdf': pdf,
                        'relevance_class': 'normal'
                    }) + '\n'
    
    yield json.dumps({
        'type': 'complete',
        'message': f'Search completed. Used keywords: {", ".join(used_keywords)}',
        'used_keywords': used_keywords,
        'total_results': len(seen_urls)
    }) + '\n'

@app.route('/search', methods=['POST'])
def search_pdfs():
    try:
        data = request.get_json()
        keyword = data.get('keyword', '')
        max_pages = min(int(data.get('max_pages', 1)), 20)
        min_relevance = float(data.get('min_relevance', 0.5))
        show_low_relevance = data.get('show_low_relevance', True)
        
        if not keyword:
            return jsonify({'error': 'Keyword is required'}), 400
            
        return Response(
            stream_search_results(keyword, max_pages, min_relevance, show_low_relevance),
            mimetype='text/event-stream'
        )
        
    except ValueError:
        return jsonify({'error': 'Invalid parameters'}), 400
    except Exception as e:
        error_message = f"Search error: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message}), 500

@app.route('/download-all', methods=['POST'])
def download_all_pdfs():
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({'error': 'No URLs provided'}), 400
            
        def download_pdf(url):
            try:
                url = unquote(url)
                filename = url.split('/')[-1]
                if not filename.lower().endswith('.pdf'):
                    filename += '.pdf'
                    
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/pdf,*/*'
                }
                
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                if response.status_code == 200:
                    content = io.BytesIO()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            content.write(chunk)
                    content.seek(0)
                    return filename, content.getvalue()
                return None, None
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")
                return None, None

        # Create a memory buffer for the ZIP file
        zip_buffer = io.BytesIO()
        
        # Download PDFs in parallel and create ZIP
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(download_pdf, url): url for url in urls}
                
                for future in as_completed(future_to_url):
                    filename, content = future.result()
                    if filename and content:
                        zip_file.writestr(filename, content)
        
        # Prepare the ZIP file for download
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='pdfs.zip'
        )
        
    except Exception as e:
        error_message = f"Download error: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message}), 500

@app.after_request
def after_request(response):
    """Enable CORS"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000) 