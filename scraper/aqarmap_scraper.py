"""
Aqarmap Egypt Real Estate Scraper - Complete Version
Scrapes: title, price, description, location, bedrooms, bathrooms, area, images
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
import re
from urllib.parse import urljoin, urlparse

class AqarmapScraper:
    def __init__(self):
        self.base_url = "https://aqarmap.com.eg"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def get_property_urls_from_search(self, search_url, max_pages=3):
        """Extract property URLs from search results page"""
        property_urls = []
        
        for page in range(1, max_pages + 1):
            page_url = f"{search_url}?page={page}"
            print(f"   🔍 Fetching page {page}: {page_url}")
            
            try:
                response = self.session.get(page_url, timeout=15)
                if response.status_code != 200:
                    print(f"   ⚠️ Status code: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all links that look like property listings
                all_links = soup.find_all('a', href=True)
                
                for link in all_links:
                    href = link.get('href')
                    if href:
                        # Check if it's a property link
                        if '/en/for-sale/' in href or '/en/ad/' in href or '/property/' in href:
                            if href.startswith('/'):
                                full_url = urljoin(self.base_url, href)
                            else:
                                full_url = href
                            
                            # Clean URL (remove query parameters)
                            full_url = full_url.split('?')[0]
                            
                            if full_url not in property_urls and self.base_url in full_url:
                                property_urls.append(full_url)
                
                print(f"      Found {len(property_urls)} unique URLs so far")
                time.sleep(2)
                
            except Exception as e:
                print(f"   Error on page {page}: {e}")
                continue
        
        return list(set(property_urls))
    
    def scrape_property(self, url):
        """Scrape detailed information from a property page"""
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            page_text = soup.get_text()
            
            # Extract Title
            title = "Unknown"
            title_tag = soup.find('h1')
            if title_tag:
                title = title_tag.text.strip()
            elif soup.find('title'):
                title = soup.find('title').text.strip()
            
            # Extract Price
            price = self._extract_price(soup, page_text)
            if price == 0:
                return None  # Skip if no price
            
            # Extract Description
            description = self._extract_description(soup)
            
            # Extract Location
            location = self._extract_location(soup, page_text)
            
            # Extract Bedrooms
            bedrooms = self._extract_bedrooms(page_text)
            
            # Extract Bathrooms
            bathrooms = self._extract_bathrooms(page_text)
            
            # Extract Area
            area = self._extract_area(page_text)
            
            # Extract Property Type
            property_type = self._detect_property_type(title, description, page_text)
            
            # Extract Furnishing
            furnishing = self._detect_furnishing(description, page_text)
            
            # Extract Images
            images = self._extract_images(soup)
            
            # Download images
            local_images = self._download_images(images, title)
            
            return {
                'url': url,
                'title': title,
                'price': price,
                'description': description,
                'location': location,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'area_sqm': area,
                'property_type': property_type,
                'furnishing': furnishing,
                'images': local_images,
                'source': 'aqarmap',
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def _extract_price(self, soup, page_text):
        """Extract price from page"""
        # Try multiple selectors
        price_selectors = [
            'span[class*="price"]',
            'div[class*="price"]',
            'strong[class*="price"]',
            '.price',
            '.Price',
            '[data-testid="price"]'
        ]
        
        for selector in price_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.text.strip()
                price = self._clean_price(text)
                if price > 0:
                    return price
        
        # Search in page text
        patterns = [
            r'([\d,]+)\s*(?:EGP|جنيه|LE|pound)',
            r'([\d,]+)\s*جنيه',
            r'price["\']?\s*:\s*["\']?([\d,]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return self._clean_price(match.group(1))
        
        return 0
    
    def _clean_price(self, text):
        """Clean price text to get integer"""
        clean = re.sub(r'[^\d]', '', str(text))
        try:
            return int(clean) if clean else 0
        except:
            return 0
    
    def _extract_description(self, soup):
        """Extract description from page"""
        desc_selectors = [
            'div[class*="description"]',
            'div[class*="details"]',
            'div[class*="about"]',
            'div[class*="overview"]',
            '.description',
            '.property-description'
        ]
        
        for selector in desc_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.text.strip()
                if len(text) > 50:
                    return text[:2000]
        
        return ""
    
    def _extract_location(self, soup, page_text):
        """Extract location from page"""
        loc_selectors = [
            'span[class*="location"]',
            'div[class*="location"]',
            'div[class*="area"]',
            '.location',
            '.address'
        ]
        
        for selector in loc_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.text.strip()
                if len(text) > 3:
                    return text[:100]
        
        # Try regex
        match = re.search(r'(?:location|area|address)[:\s]+([A-Za-z\s]+)', page_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:100]
        
        return "Cairo"
    
    def _extract_bedrooms(self, page_text):
        """Extract number of bedrooms"""
        patterns = [
            r'(\d+)\s*(?:bed|bedroom|bedrooms|غرفة|غرف)',
            r'(\d+)\s*(?:BR|br)',
            r'bedrooms?["\']?\s*:\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0
    
    def _extract_bathrooms(self, page_text):
        """Extract number of bathrooms"""
        patterns = [
            r'(\d+)\s*(?:bath|bathroom|bathrooms|حمام|حمامات)',
            r'(\d+)\s*(?:BA|ba)',
            r'bathrooms?["\']?\s*:\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 1
    
    def _extract_area(self, page_text):
        """Extract area in square meters"""
        patterns = [
            r'(\d+)\s*(?:m²|sqm|sq\.?m\.?|مساحة|متر)',
            r'(\d+)\s*(?:square meter|square meters)',
            r'area["\']?\s*:\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0
    
    def _detect_property_type(self, title, description, page_text):
        """Detect property type from text"""
        text = (title + " " + description + " " + page_text).lower()
        
        if 'villa' in text or 'فيلا' in text:
            return 'Villa'
        elif 'townhouse' in text or 'تاون هاوس' in text:
            return 'Townhouse'
        elif 'studio' in text or 'ستوديو' in text:
            return 'Studio'
        elif 'penthouse' in text or 'بنتهاوس' in text:
            return 'Penthouse'
        elif 'duplex' in text or 'دوبلكس' in text:
            return 'Duplex'
        else:
            return 'Apartment'
    
    def _detect_furnishing(self, description, page_text):
        """Detect furnishing status"""
        text = (description + " " + page_text).lower()
        
        if 'furnished' in text or 'مفروش' in text:
            return 'Furnished'
        elif 'semi-furnished' in text or 'نصف مفروش' in text:
            return 'Semi-furnished'
        else:
            return 'Unfurnished'
    
    def _extract_images(self, soup):
        """Extract image URLs from page"""
        images = []
        
        # Find all img tags
        img_tags = soup.find_all('img')
        
        for img in img_tags:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src and src.startswith('http'):
                # Filter out small icons and logos
                if 'logo' not in src.lower() and 'icon' not in src.lower():
                    images.append(src)
        
        # Remove duplicates
        return list(set(images))
    
    def _download_images(self, urls, title, max_images=5):
        """Download images locally"""
        local_paths = []
        
        if not urls:
            return []
        
        # Create safe directory name
        safe_title = re.sub(r'[^\w\-_]', '_', title)[:50]
        img_dir = f"data/raw/images/{safe_title}"
        os.makedirs(img_dir, exist_ok=True)
        
        for i, url in enumerate(urls[:max_images]):
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    # Determine file extension
                    content_type = response.headers.get('content-type', '')
                    if 'png' in content_type:
                        ext = 'png'
                    elif 'webp' in content_type:
                        ext = 'webp'
                    else:
                        ext = 'jpg'
                    
                    filepath = os.path.join(img_dir, f"img_{i}.{ext}")
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    local_paths.append(filepath)
                    print(f"      📸 Downloaded image {i+1}/{min(len(urls), max_images)}")
            except Exception as e:
                print(f"      ⚠️ Failed: {e}")
            
            time.sleep(0.5)
        
        return local_paths
    
    def scrape_all(self, search_urls, max_pages_per_search=2, max_properties=100):
        """Main scraping function"""
        all_properties = []
        all_urls = []
        
        print("\n" + "="*70)
        print("🔍 STEP 1: Collecting property URLs")
        print("="*70)
        
        for search_url in search_urls:
            print(f"\n📂 Searching: {search_url}")
            urls = self.get_property_urls_from_search(search_url, max_pages_per_search)
            all_urls.extend(urls)
            print(f"   Total URLs so far: {len(all_urls)}")
        
        # Remove duplicates
        all_urls = list(set(all_urls))
        print(f"\n📊 Unique property URLs found: {len(all_urls)}")
        
        # Limit
        if max_properties:
            all_urls = all_urls[:max_properties]
        
        print("\n" + "="*70)
        print("📄 STEP 2: Scraping property details")
        print("="*70)
        
        for idx, url in enumerate(all_urls):
            print(f"\n📌 [{idx+1}/{len(all_urls)}] Scraping: {url[:80]}...")
            
            property_data = self.scrape_property(url)
            
            if property_data and property_data['price'] > 0:
                all_properties.append(property_data)
                print(f"   ✅ {property_data['title'][:50]}")
                print(f"   💰 {property_data['price']:,.0f} EGP")
                print(f"   🛏️ {property_data['bedrooms']} beds | 🚽 {property_data['bathrooms']} baths")
                print(f"   📐 {property_data['area_sqm']} sqm | 🏠 {property_data['property_type']}")
                print(f"   📸 {len(property_data['images'])} images")
            else:
                print(f"   ❌ Failed (no price)")
            
            time.sleep(1.5)  # Be respectful
        
        return all_properties
    
    def save_data(self, properties, filename="aqarmap_properties.json"):
        """Save scraped data to JSON"""
        os.makedirs('data/raw', exist_ok=True)
        filepath = os.path.join('data/raw', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(properties, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Saved {len(properties)} properties to {filepath}")
        return filepath


if __name__ == "__main__":
    print("="*70)
    print("🏠 AQARMAP EGYPT REAL ESTATE SCRAPER")
    print("="*70)
    
    scraper = AqarmapScraper()
    
    # Search URLs (add more cities as needed)
    search_urls = [
        "https://aqarmap.com.eg/en/for-sale/apartment/cairo/",
        "https://aqarmap.com.eg/en/for-sale/villa/cairo/",
        "https://aqarmap.com.eg/en/for-sale/apartment/alexandria/",
        "https://aqarmap.com.eg/en/for-sale/apartment/giza/",
        "https://aqarmap.com.eg/en/for-sale/townhouse/cairo/",
        "https://aqarmap.com.eg/en/for-sale/studio/cairo/",
    ]
    
    # Scrape
    properties = scraper.scrape_all(
        search_urls=search_urls,
        max_pages_per_search=2,  # Increase for more data
        max_properties=100        # Maximum properties to scrape
    )
    
    # Save
    if properties:
        scraper.save_data(properties)
        
        # Statistics
        print("\n" + "="*70)
        print("📊 SCRAPING STATISTICS")
        print("="*70)
        print(f"✅ Total properties: {len(properties)}")
        
        prices = [p['price'] for p in properties]
        print(f"💰 Price range: {min(prices):,.0f} - {max(prices):,.0f} EGP")
        print(f"📊 Average price: {sum(prices)/len(prices):,.0f} EGP")
        
        # By type
        type_counts = {}
        for p in properties:
            t = p['property_type']
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"🏠 Property types: {type_counts}")
        
        # By furnishing
        furn_counts = {}
        for p in properties:
            f = p['furnishing']
            furn_counts[f] = furn_counts.get(f, 0) + 1
        print(f"🪑 Furnishing: {furn_counts}")
        
        # Images
        total_images = sum(len(p['images']) for p in properties)
        print(f"📸 Total images downloaded: {total_images}")
        
    else:
        print("\n❌ No properties scraped!")
        print("   Possible reasons:")
        print("   1. Aqarmap changed their HTML structure")
        print("   2. Network issues")
        print("   3. The website is blocking requests")
        
        print("\n📝 Creating sample data for testing...")
        # Create sample data
        sample_data = []
        for i in range(50):
            sample_data.append({
                'url': f"https://aqarmap.com.eg/property/{i}",
                'title': f"Sample Property {i}",
                'price': 1000000 + (i * 100000),
                'description': "This is a sample property for testing purposes.",
                'location': "Cairo",
                'bedrooms': (i % 4) + 1,
                'bathrooms': (i % 3) + 1,
                'area_sqm': 80 + (i % 120),
                'property_type': ['Apartment', 'Villa', 'Studio'][i % 3],
                'furnishing': ['Furnished', 'Semi-furnished', 'Unfurnished'][i % 3],
                'images': [],
                'source': 'sample',
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        scraper.save_data(sample_data, "aqarmap_properties.json")