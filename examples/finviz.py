#!/usr/bin/env python3
"""
Finviz Performance Data Scraper
This script scrapes performance data from Finviz and exports it to CSV
"""

import pandas as pd
import time
import logging
from finvizfinance.screener.overview import Overview
from finvizfinance.screener.performance import Performance
from finvizfinance.quote import Quote
from typing import List, Dict, Optional
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finviz_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FinvizPerformanceScraper:
    """Class to scrape performance data from Finviz"""
    
    def __init__(self, sleep_sec: float = 1.0):
        """
        Initialize the scraper
        
        Args:
            sleep_sec: Delay between requests to avoid rate limiting
        """
        self.sleep_sec = sleep_sec
        self.screener = Performance()
        
    def get_all_tickers(self, limit: int = 5) -> List[str]:
        """
        Get all tickers from Finviz screener
        
        Args:
            limit: Maximum number of tickers to retrieve
            
        Returns:
            List of ticker symbols
        """
        try:
            logger.info("Fetching all tickers from Finviz screener...")
            
            # Get screener data with specified parameters
            dfs = []
            for page in range(1, 102):
                screener_data = self.screener.screener_view(
                    order='Ticker',
                    limit=limit,
                    select_page=page,
                    verbose=1,
                    ascend=True,
                    columns=None,
                    sleep_sec=self.sleep_sec
                )
                dfs.append(screener_data)

            df = pd.concat(dfs)

            df.to_csv("/home/ubuntu/hyperliquid-python-sdk/examples/perf.csv", index=False)

            import pdb; pdb.set_trace();
            
            # if screener_data is not None and not screener_data.empty:
            #     tickers = screener_data['Ticker'].tolist()
            #     logger.info(f"Retrieved {len(tickers)} tickers")
            #     return tickers
            # else:
            #     logger.warning("No ticker data retrieved")
            #     return []
                
        except Exception as e:
            logger.error(f"Error fetching tickers: {str(e)}")
            return []
    
    def get_ticker_performance(self, ticker: str) -> Optional[Dict]:
        """
        Get performance data for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing performance data or None if error
        """
        try:
            quote = Quote(ticker)

            
            # Get performance data
            performance_data = quote.ticker_performance()
            
            if performance_data:
                # Add ticker symbol to the data
                performance_data['Ticker'] = ticker
                logger.debug(f"Retrieved performance data for {ticker}")
                return performance_data
            else:
                logger.warning(f"No performance data available for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting performance data for {ticker}: {str(e)}")
            return None
    
    def scrape_all_performance_data(self, tickers: List[str] = None, 
                                  max_tickers: Optional[int] = None) -> pd.DataFrame:
        """
        Scrape performance data for all tickers
        
        Args:
            tickers: List of specific tickers to scrape. If None, gets all tickers
            max_tickers: Maximum number of tickers to process (for testing)
            
        Returns:
            DataFrame containing all performance data
        """
        if tickers is None:
            tickers = self.get_all_tickers()
        
        if max_tickers:
            tickers = tickers[:max_tickers]
        
        logger.info(f"Starting to scrape performance data for {len(tickers)} tickers...")
        
        all_performance_data = []
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing ticker {i}/{len(tickers)}: {ticker}")
            
            performance_data = self.get_ticker_performance(ticker)
            
            if performance_data:
                all_performance_data.append(performance_data)
            
            # Sleep to avoid rate limiting
            if i < len(tickers):  # Don't sleep after the last ticker
                time.sleep(self.sleep_sec)
        
        if all_performance_data:
            df = pd.DataFrame(all_performance_data)
            logger.info(f"Successfully scraped performance data for {len(df)} tickers")
            return df
        else:
            logger.warning("No performance data was successfully scraped")
            return pd.DataFrame()
    
    def export_to_csv(self, df: pd.DataFrame, filename: str = 'finviz_performance_data.csv'):
        """
        Export DataFrame to CSV
        
        Args:
            df: DataFrame to export
            filename: Output filename
        """
        try:
            if not df.empty:
                df.to_csv(filename, index=False)
                logger.info(f"Data exported to {filename}")
                logger.info(f"Exported {len(df)} rows and {len(df.columns)} columns")
            else:
                logger.warning("No data to export")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")

def main():
    """Main function to run the scraper"""
    
    # Configuration
    SLEEP_SEC = 0.0001  # Delay between requests
    MAX_TICKERS = 5  # Set to a number for testing, None for all tickers
    OUTPUT_FILE = 'finviz_performance_data.csv'
    
    # Initialize scraper
    scraper = FinvizPerformanceScraper(sleep_sec=SLEEP_SEC)
    
    try:
        # Option 1: Scrape all tickers
        logger.info("Starting Finviz performance data scraping...")
        performance_df = scraper.scrape_all_performance_data(max_tickers=MAX_TICKERS)
        
        # Export to CSV
        scraper.export_to_csv(performance_df, OUTPUT_FILE)
        
        # Display summary
        if not performance_df.empty:
            logger.info("=== SCRAPING SUMMARY ===")
            logger.info(f"Total tickers processed: {len(performance_df)}")
            logger.info(f"Columns in dataset: {list(performance_df.columns)}")
            logger.info(f"Output file: {OUTPUT_FILE}")
            
            # Show first few rows
            print("\nFirst 5 rows of data:")
            print(performance_df.head())
        else:
            logger.error("No data was scraped successfully")
    
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")

if __name__ == "__main__":
    main()