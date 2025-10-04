# scraperbackend.py
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import google.genai as genai
import re
import time
from urllib.parse import urlparse
import logging
import random
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configuration constants
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]
TIMEOUT = 30
MAX_CONTENT_LENGTH = 50000

# Create a global session for reuse
session = requests.Session()
session.headers.update({
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
})

def is_valid_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_random_user_agent() -> str:
    """Get a random user agent"""
    return random.choice(USER_AGENTS)

def extract_title(soup: BeautifulSoup) -> str:
    """Extract page title from HTML"""
    title_selectors = [
        'h1',
        'title',
        '.article-title',
        '.post-title',
        '.entry-title',
        '[class*="headline"]',
        '[class*="title"]'
    ]
    
    for selector in title_selectors:
        element = soup.select_one(selector)
        if element and element.get_text().strip():
            return element.get_text().strip()
    
    return "Untitled"

def remove_unwanted_elements(soup: BeautifulSoup):
    """Remove unwanted HTML elements"""
    unwanted_tags = [
        'script', 'style', 'nav',
        'header', 'footer', 'aside',
        'advertisement', 'title'
    ]
    
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()

def clean_text(text: str) -> str: # Clean and format extracted text
    # Remove extra whitespace
    text = re.sub(r'\n+', '\n', text)
    
    # Split into lines and clean
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()
        # Keep lines that are substantial (not just navigation elements)
        if len(line) > 20 and not line.lower().startswith(('menu', 'navigation', 'skip to')):
            clean_lines.append(line)
    
    # Delete lines that are metadata of images
    clean_lines = [line for line in clean_lines if "image" not in line.lower() and "file" not in line.lower() and "reuters" not in line.lower() and "cnn" not in line.lower() and "ap" not in line.lower() and "@" not in line.lower()]
    
    return '. '.join(clean_lines)

def extract_content_from_soup(soup: BeautifulSoup, url: str) -> Dict[str, str]:
    """Extract title and main content from parsed HTML"""
    # Extract title
    title = extract_title(soup)
    
    # Remove unwanted elements
    remove_unwanted_elements(soup)
    
    # Clean and format text
    text = clean_text(soup.get_text(separator="\n"))
    
    # Limit content length
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH] + "... [Content truncated]"
    
    return {
        'title': title,
        'text': text
    }

def scrape_url(url: str): # Scrape content from a URL and return structured data
    try:
        # Validate URL
        if not is_valid_url(url):
            return {'success': False, 'error': 'Invalid URL format'}

        # Set random user agent
        session_headers = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}
        
        logger.info(f"Scraping URL: {url}")
        
        # Make request
        response = session.get(url, headers=session_headers, timeout=TIMEOUT)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html5lib')
        
        # Extract content
        extracted_data = extract_content_from_soup(soup, url)
        
        return {
            'success': True,
            'url': url,
            'title': extracted_data['title'],
            'text': extracted_data['text'],
            'word_count': len(extracted_data['text'].split()),
            'char_count': len(extracted_data['text'])
        }
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out'}
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': 'Connection error - check your internet or URL'}
    except requests.exceptions.HTTPError as e:
        return {'success': False, 'error': f'HTTP error: {e.response.status_code}'}
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        return {'success': False, 'error': f'Scraping failed: {str(e)}'}

def build_prompt(text: str, summary_type: str, max_words: Optional[int] = None, bullet_count: Optional[int] = None) -> str: # Build summarization prompt based on parameters
    prompts = {
        'brief': 'Summarize the following text in exactly 2-3 clear sentences:',
        'medium': 'Summarize the following text in one concise paragraph:',
        'detailed': 'Provide a detailed summary of the following text in 2-3 paragraphs:',
        'bullets': 'Summarize the following text as bullet points with key takeaways:'
    }
    
    prompt = prompts.get(summary_type, prompts['medium'])
    
    # Handle bullet points with specific count
    if summary_type == 'bullets' and bullet_count:
        prompt = f"Summarize the following text as {bullet_count} bullet points separated with 1 line breaks between each point with key takeaways: "

    # Add word limit only for non-bullet summaries
    if max_words and summary_type != 'bullets':
        prompt += f" Keep it under {max_words} words."
    
    return f"{prompt}\n\n{text}"

def summarize_with_gemini(text: str, api_key: str, summary_type: str = 'medium', max_words: Optional[int] = None, bullet_count: Optional[int] = None) -> Dict[str, Any]: # Summarize text using Gemini API
    try:
        # Configure Gemini
        client = genai.Client(api_key=api_key)
        
        # Build prompt
        prompt = build_prompt(text, summary_type, max_words)
        
        # Generate summary
        response = client.models.generate_content(model='gemini-1.5-flash', contents=prompt)
        
        if response.text:
            return {
                'success': True,
                'summary': response.text,
                'original_length': len(text.split()),
                'summary_length': len(response.text.strip().split()),
                'summary_type': summary_type,
                'bullet_count': bullet_count if summary_type == 'bullets' else None
            }
        else:
            return {'success': False, 'error': 'No summary generated'}
            
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return {'success': False, 'error': f'Summarization failed: {str(e)}'}

def get_api_docs_html() -> str: # Return HTML for API documentation
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Web Scraper API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
            pre { background: #333; color: #fff; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🕷️ Web Scraper Backend API</h1>
        <p>A Python Flask backend for web scraping and AI summarization.</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /scrape</h3>
            <p>Scrape content from a URL</p>
            <pre>
{
    "url": "https://example.com/article"
}
            </pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /summarize</h3>
            <p>Scrape and summarize content in one step</p>
            <pre>
{
    "url": "https://example.com/article",
    "api_key": "your-gemini-api-key",
    "summary_type": "medium",
    "max_words": 100
}
            </pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check if the server is running</p>
        </div>
        
        <h2>Status:</h2>
        <p>✅ Server is running on port 5000</p>
        <p>🌐 CORS enabled for all origins</p>
        <p>📝 Ready to scrape and summarize!</p>
    </body>
    </html>
    """

# Routes
@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    return render_template_string(get_api_docs_html())

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Web scraper backend is running',
        'timestamp': time.time()
    })

@app.route('/scrape', methods=['POST'])
def scrape_endpoint():
    """Scrape content from a URL"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing URL in request body'
            }), 400
        
        url = data['url']
        
        # Scrape the URL
        result = scrape_url(url)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Scrape endpoint error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/summarize', methods=['POST'])
def summarize_endpoint():
    """Scrape URL and summarize content"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Missing request body'
            }), 400
        
        # Required fields
        url = data.get('url')
        api_key = data.get('api_key')
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'Missing URL'
            }), 400
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'Missing Gemini API key'
            }), 400
        
        # Optional fields
        summary_type = data.get('summary_type', 'medium')
        max_words = data.get('max_words')
        bullet_count = data.get('bullet_count')

        # Convert string numbers to integers if provided
        if max_words:
            try:
                max_words = int(max_words)
            except (ValueError, TypeError):
                max_words = None

        if bullet_count:
            try:
                bullet_count = int(bullet_count)
            except (ValueError, TypeError):
                bullet_count = None
        
        # Scrape content
        scrape_result = scrape_url(url)
        
        if not scrape_result['success']:
            return jsonify(scrape_result)
        
        # Summarize content
        summary_result = summarize_with_gemini(
            scrape_result['text'],
            api_key,
            summary_type,
            max_words,
            bullet_count  # Pass bullet count to summarization function
        )
        
        if summary_result['success']:
            return jsonify({
                'success': True,
                'url': url,
                'title': scrape_result['title'],
                'original_text': scrape_result['text'],
                'summary': summary_result['summary'],
                'summary_type': summary_result.get('summary_type'),
                'bullet_count': summary_result.get('bullet_count'),
                'word_counts': {
                    'original': summary_result['original_length'],
                    'summary': summary_result['summary_length']
                }
            })
        else:
            return jsonify(summary_result)
        
    except Exception as e:
        logger.error(f"Summarize endpoint error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Web Scraper Backend...")
    print("📍 Server will run at: http://localhost:5000")
    print("📖 API Documentation: http://localhost:5000")
    print("🔍 Test scraping: POST to /scrape with {'url': 'https://example.com'}")
    print("✨ Test summarization: POST to /summarize with url and api_key")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )