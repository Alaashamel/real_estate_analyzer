"""
Main execution script
Run: python main.py --scrape --train --app
"""

import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description='Real Estate AI System')
    parser.add_argument('--scrape', action='store_true', help='Run web scrapers')
    parser.add_argument('--train', action='store_true', help='Train multimodal model')
    parser.add_argument('--app', action='store_true', help='Launch Streamlit app')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    
    args = parser.parse_args()
    
    if args.all or args.scrape:
        print("🕷️ Running scrapers...")
        from scraper.olx_scraper import OLXEgyptScraper
        scraper = OLXEgyptScraper()
        properties = scraper.scrape_multiple(queries=['apartment', 'villa'], max_pages=2)
        
        print("🔄 Running data pipeline...")
        from scraper.data_pipeline import DataPipeline
        pipeline = DataPipeline()
        df = pipeline.run_full_pipeline()
    
    if args.all or args.train:
        print("🏋️ Training multimodal model...")
        from multimodal.price_predictor import CompletePricePredictor
        import pandas as pd
        
        df = pd.read_csv('data/processed/final_dataset.csv')
        predictor = CompletePricePredictor()
        predictor.train(df, epochs=30)
        predictor.save()
    
    if args.all or args.app:
        print("🚀 Launching Streamlit app...")
        subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

if __name__ == "__main__":
    main()