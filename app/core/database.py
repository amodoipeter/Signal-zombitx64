from supabase import create_client, Client
from typing import Optional
from functools import lru_cache

from app.core.config import settings

@lru_cache()
def get_supabase() -> Client:
    """
    Get or create Supabase client instance.
    Uses lru_cache to maintain a single instance.
    """
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

class Database:
    def __init__(self):
        """Initialize Supabase client."""
        self.client = get_supabase()

    async def execute(self, table: str, query_type: str, data: dict = None,
                     filters: dict = None) -> Optional[dict]:
        """
        Execute database operations.
        
        Args:
            table: Table name
            query_type: Type of query ('select', 'insert', 'update', 'delete')
            data: Data for insert/update operations
            filters: Filter conditions for select/update/delete operations
        """
        try:
            query = self.client.table(table)

            if query_type == 'select':
                if filters:
                    for key, value in filters.items():
                        query = query.eq(key, value)
                return query.execute()

            elif query_type == 'insert':
                return query.insert(data).execute()

            elif query_type == 'update':
                if filters:
                    for key, value in filters.items():
                        query = query.eq(key, value)
                return query.update(data).execute()

            elif query_type == 'delete':
                if filters:
                    for key, value in filters.items():
                        query = query.eq(key, value)
                return query.delete().execute()

        except Exception as e:
            print(f"Database error: {str(e)}")
            raise

# Create database instance
db = Database()
