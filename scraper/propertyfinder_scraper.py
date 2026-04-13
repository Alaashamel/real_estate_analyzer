"""
Property Finder Scraper using Apify API
Alternative approach - uses Apify's ready-made scraper
"""

import requests
import json
import os
import time
from typing import List, Dict, Optional

class PropertyFinderScraper:
    """
    Scraper for Property Finder using Apify API
    Note: You need an Apify API token (free tier available)
    """
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the scraper
        
        Args:
            api_token: Apify API token (get from https://console.apify.com)
                      If not provided, will try to read from environment
        """
        self.api_token = api_token or os.environ.get('APIFY_API_TOKEN')
        self.base_url = "https://api.apify.com/v2"
        
        if not self.api_token:
            print("⚠️ No Apify API token found. Using direct HTTP requests fallback.")
            self.use_api = False
        else:
            self.use_api = True
    
    def scrape_via_apify(self, country="eg", max_results=50):
        """
        Use Apify's Property Finder scraper actor
        
        Args:
            country: Country code (eg for Egypt)
            max_results: Maximum number of results to scrape
        """
        if not self.use_api:
            print("❌ Apify API not configured. Please set APIFY_API_TOKEN")
            return []
        
        # Apify actor ID for Property Finder scraper
        actor_id = "dz_omar/propertyfinder-scraper"
        
        # Prepare input
        run_input = {
            "startUrls": [
                f"https://www.propertyfinder.{country}/en/search?c=3&fu=0&ob=mr"
            ],
            "maxResults": max_results,
            "proxyConfiguration": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"]
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        try:
            # Run the actor
            run_response = requests.post(
                f"{self.base_url}/acts/{actor_id}/runs",
                json=run_input,
                headers=headers
            )
            run_response.raise_for_status()
            run_data = run_response.json()
            run_id = run_data['data']['id']
            
            print(f"🚀 Apify actor started. Run ID: {run_id}")
            
            # Wait for completion
            while True:
                status_response = requests.get(
                    f"{self.base_url}/acts/{actor_id}/runs/{run_id}",
                    headers=headers
                )
                status = status_response.json()['data']['status']
                
                if status == 'SUCCEEDED':
                    print("✅ Scraping completed successfully!")
                    break
                elif status in ['FAILED', 'TIMED-OUT', 'ABORTED']:
                    print(f"❌ Scraping failed with status: {status}")
                    return []
                
                print(f"⏳ Status: {status}. Waiting...")
                time.sleep(5)
            
            # Fetch results
            dataset_response = requests.get(
                f"{self.base_url}/acts/{actor_id}/runs/{run_id}/dataset/items",
                headers=headers
            )
            items = dataset_response.json()
            
            # Process and save
            processed = self._process_apify_results(items)
            self._save_results(processed, "propertyfinder_results.json")
            
            return processed
            
        except Exception as e:
            print(f"❌ Apify API error: {e}")
            return []
    
    def scrape_direct_http(self, search_url: str) -> List[Dict]:
        """
        Fallback method: direct HTTP scraping with BeautifulSoup
        """
        import requests
        from bs4 import BeautifulSoup
        
        properties = []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find property cards (selectors may change - needs updating)
            cards = soup.find_all('div', class_=re.compile(r'card|property|listing'))
            
            for card in cards:
                property_data = self._extract_from_card(card)
                if property_data:
                    properties.append(property_data)
            
        except Exception as e:
            print(f"Error scraping: {e}")
        
        return properties
    
    def _extract_from_card(self, card):
        """Extract property data from a card element"""
        # This method needs to be updated based on current website structure
        # Placeholder implementation
        return {
            'title': "Property Title",
            'price': 0,
            'description': "",
            'images': []
        }
    
    def _process_apify_results(self, items):
        """Process results from Apify into standard format"""
        processed = []
        
        for item in items:
            property_data = {
                'url': item.get('url', ''),
                'title': item.get('title', ''),
                'price': self._extract_price(item.get('price', '0')),
                'description': item.get('description', ''),
                'bedrooms': item.get('bedrooms', 0),
                'bathrooms': item.get('bathrooms', 0),
                'area_sqm': item.get('area', 0),
                'property_type': item.get('property_type', 'Apartment'),
                'furnishing': item.get('furnishing', 'Unknown'),
                'location': item.get('location', ''),
                'images': item.get('images', []),
                'source': 'propertyfinder',
                'agent_name': item.get('agent_name', ''),
                'agent_phone': item.get('agent_phone', '')
            }
            processed.append(property_data)
        
        return processed
    
    def _extract_price(self, price_text):
        """Extract numeric price from text"""
        import re
        if isinstance(price_text, (int, float)):
            return float(price_text)
        
        clean = re.sub(r'[^\d.]', '', str(price_text).replace(',', ''))
        try:
            return float(clean) if clean else 0
        except:
            return 0
    
    def _save_results(self, properties, filename):
        """Save scraped data to JSON"""
        os.makedirs('data/raw', exist_ok=True)
        filepath = os.path.join('data/raw', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(properties, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved {len(properties)} properties to {filepath}")


if __name__ == "__main__":
    # Example: Direct HTTP scraping (no API key needed for basic scraping)
    scraper = PropertyFinderScraper(api_token=None)
    
    # Try direct scraping
    print("🔍 Attempting direct scraping...")
    properties = scraper.scrape_direct_http("https://www.propertyfinder.eg/en/search?c=3")
    
    print(f"Scraped {len(properties)} properties")