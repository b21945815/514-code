import sqlite3
import pandas as pd
import os

class BirdDBReader:
    def __init__(self, db_filename='financial.sqlite'):
        """
        Initializes the database path.
        The connection is not established immediately; it opens when entering the 'with' block.
        """
        self.db_path = self._find_database(db_filename)
        self.conn = None
        
    def _find_database(self, filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        candidate_path = os.path.join(base_dir, filename)
        if os.path.exists(candidate_path):
            return candidate_path
    
        for root, dirs, files in os.walk(base_dir):
            if filename in files:
                found_path = os.path.join(root, filename)
                print(f"File: {found_path}")
                return found_path
        
        raise FileNotFoundError("SQL FILE NOT FOUND")

    def __enter__(self):
        """
        Context Manager Entry.
        Opens the database connection in READ-ONLY mode.
        """
        try:
            # The 'uri=True' parameter and '?mode=ro' query string are critical here.
            # This prevents any write attempts at the OS level.
            uri_path = f"file:{os.path.abspath(self.db_path)}?mode=ro"
            self.conn = sqlite3.connect(uri_path, uri=True)
            return self
        except sqlite3.Error as e:
            print(f"Connection error: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context Manager Exit.
        Safely closes the connection whether an error occurred or not.
        """
        if self.conn:
            self.conn.close()

    def run_select_query(self, sql_query):
        """
        Executes only SELECT queries and returns a Pandas DataFrame.
        """
        if not self.conn:
            raise ConnectionError("Connection is not open. Please use the 'with' block.")
        
        try:
            return pd.read_sql_query(sql_query, self.conn)
        except Exception as e:
            print(f"Query error: {e}")
            return None