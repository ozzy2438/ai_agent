import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

class Database:
    """Database operations class"""
    
    @staticmethod
    async def sign_in_with_email(email: str, password: str) -> Optional[Dict[str, Any]]:
        """Sign in user with email and password using Supabase Auth"""
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return response.user if response else None
        except Exception as e:
            print(f"Error signing in: {str(e)}")
            return None

    @staticmethod
    async def get_user(user_id: str) -> Optional[Dict[str, Any]]:
        """Get user details from Supabase"""
        try:
            response = supabase.from_("users").select("*").eq("id", user_id).single().execute()
            return response.data if response else None
        except Exception as e:
            print(f"Error getting user: {str(e)}")
            return None

    @staticmethod
    async def save_document(
        filename: str,
        content: str,
        metadata: Dict[str, Any],
        collection_name: str
    ) -> Dict[str, Any]:
        """Save document to database"""
        try:
            data = {
                "filename": filename,
                "content": content,
                "metadata": metadata,
                "collection_name": collection_name,
                "created_at": datetime.now().isoformat()
            }
            result = supabase.table("documents").insert(data).execute()
            return result.data[0]
        except Exception as e:
            print(f"Error saving document: {str(e)}")
            return {}

    @staticmethod
    async def save_query(
        query: str,
        answer: str,
        source_documents: List[Dict[str, Any]],
        token_usage: Dict[str, int],
        total_cost: float
    ) -> Dict[str, Any]:
        """Save query to database"""
        try:
            data = {
                "query": query,
                "answer": answer,
                "source_documents": source_documents,
                "token_usage": token_usage,
                "total_cost": total_cost,
                "created_at": datetime.now().isoformat()
            }
            result = supabase.table("queries").insert(data).execute()
            return result.data[0]
        except Exception as e:
            print(f"Error saving query: {str(e)}")
            return {}

    @staticmethod
    async def get_analytics() -> Dict[str, Any]:
        """Get analytics data"""
        try:
            # Get all queries
            queries = supabase.table("queries").select("*").execute()
            
            # Calculate metrics
            total_queries = len(queries.data)
            total_cost = sum(q["total_cost"] for q in queries.data)
            average_tokens = sum(q["token_usage"]["total_tokens"] for q in queries.data) / total_queries if total_queries > 0 else 0
            
            return {
                "total_queries": total_queries,
                "total_cost": total_cost,
                "average_tokens": average_tokens,
                "queries_by_date": queries.data
            }
        except Exception as e:
            print(f"Error getting analytics: {str(e)}")
            return {
                "total_queries": 0,
                "total_cost": 0,
                "average_tokens": 0,
                "queries_by_date": []
            }

    @staticmethod
    async def get_query_history() -> List[Dict[str, Any]]:
        """Get query history"""
        try:
            result = supabase.table("queries").select("*").order("created_at", desc=True).limit(10).execute()
            return result.data
        except Exception as e:
            print(f"Error getting query history: {str(e)}")
            return []

    @staticmethod
    async def validate_user(email: str, password_hash: str) -> Optional[Dict[str, Any]]:
        """Validate user credentials"""
        try:
            result = supabase.table("users").select("*").eq("email", email).eq("password_hash", password_hash).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error validating user: {str(e)}")
            return None 