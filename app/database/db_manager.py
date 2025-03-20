"""
Database Manager module for the AI Signal Provider.
This module handles database connections and operations.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create base class for SQLAlchemy models
Base = declarative_base()

# Define SQLAlchemy models
class User(Base):
    """User model for storing user related data."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class Subscription(Base):
    """Subscription model for storing subscription related data."""
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    plan_name = Column(String(50), nullable=False)
    stripe_customer_id = Column(String(100))
    stripe_subscription_id = Column(String(100))
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    
    def __repr__(self):
        return f"<Subscription(id={self.id}, user_id={self.user_id}, plan='{self.plan_name}')>"


class Signal(Base):
    """Signal model for storing trading signal data."""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY, SELL, NEUTRAL
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    outcomes = relationship("SignalOutcome", back_populates="signal")
    
    def __repr__(self):
        return f"<Signal(id={self.id}, symbol='{self.symbol}', type='{self.signal_type}')>"


class SignalOutcome(Base):
    """SignalOutcome model for storing the outcomes of trading signals."""
    __tablename__ = "signal_outcomes"
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=False)
    exit_price = Column(Float, nullable=False)
    exit_timestamp = Column(DateTime, nullable=False)
    return_pct = Column(Float, nullable=False)
    is_successful = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    signal = relationship("Signal", back_populates="outcomes")
    
    def __repr__(self):
        return f"<SignalOutcome(id={self.id}, signal_id={self.signal_id}, return={self.return_pct:.2f}%)>"


class MarketData(Base):
    """MarketData model for storing historical market data."""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timestamp='{self.timestamp}')>"


class DBManager:
    """
    Database manager class for handling database operations.
    """
    
    def __init__(self):
        """
        Initialize the DBManager.
        """
        # Get database URL from environment variables
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("Database URL not found in environment variables")
        
        # Create engine and session
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        logger.info("Database manager initialized")
    
    def create_tables(self):
        """
        Create all database tables.
        """
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def drop_tables(self):
        """
        Drop all database tables.
        """
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping database tables: {str(e)}")
            raise
    
    # User operations
    def create_user(self, user_data: Dict[str, Any]) -> User:
        """
        Create a new user.
        
        Args:
            user_data (dict): User data
            
        Returns:
            User: Created user object
        """
        try:
            session = self.Session()
            user = User(**user_data)
            session.add(user)
            session.commit()
            user_id = user.id
            session.close()
            
            logger.info(f"Created user with ID {user_id}")
            return user
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            if session:
                session.rollback()
                session.close()
            raise
    
    def get_user(self, user_id: int) -> Optional[User]:
        """
        Get a user by ID.
        
        Args:
            user_id (int): User ID
            
        Returns:
            User: User object if found, None otherwise
        """
        try:
            session = self.Session()
            user = session.query(User).filter(User.id == user_id).first()
            session.close()
            
            return user
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {str(e)}")
            if session:
                session.close()
            raise
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email.
        
        Args:
            email (str): User email
            
        Returns:
            User: User object if found, None otherwise
        """
        try:
            session = self.Session()
            user = session.query(User).filter(User.email == email).first()
            session.close()
            
            return user
        except Exception as e:
            logger.error(f"Error getting user by email {email}: {str(e)}")
            if session:
                session.close()
            raise
    
    # Subscription operations
    def create_subscription(self, subscription_data: Dict[str, Any]) -> Subscription:
        """
        Create a new subscription.
        
        Args:
            subscription_data (dict): Subscription data
            
        Returns:
            Subscription: Created subscription object
        """
        try:
            session = self.Session()
            subscription = Subscription(**subscription_data)
            session.add(subscription)
            session.commit()
            subscription_id = subscription.id
            session.close()
            
            logger.info(f"Created subscription with ID {subscription_id}")
            return subscription
        except Exception as e:
            logger.error(f"Error creating subscription: {str(e)}")
            if session:
                session.rollback()
                session.close()
            raise
    
    def get_user_subscriptions(self, user_id: int) -> List[Subscription]:
        """
        Get all subscriptions for a user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            list: List of subscription objects
        """
        try:
            session = self.Session()
            subscriptions = session.query(Subscription).filter(Subscription.user_id == user_id).all()
            session.close()
            
            return subscriptions
        except Exception as e:
            logger.error(f"Error getting subscriptions for user {user_id}: {str(e)}")
            if session:
                session.close()
            raise
    
    def update_subscription(self, subscription_id: int, updates: Dict[str, Any]) -> Optional[Subscription]:
        """
        Update a subscription.
        
        Args:
            subscription_id (int): Subscription ID
            updates (dict): Updates to apply
            
        Returns:
            Subscription: Updated subscription object
        """
        try:
            session = self.Session()
            subscription = session.query(Subscription).filter(Subscription.id == subscription_id).first()
            
            if not subscription:
                session.close()
                return None
            
            for key, value in updates.items():
                setattr(subscription, key, value)
            
            session.commit()
            updated_subscription = session.query(Subscription).filter(Subscription.id == subscription_id).first()
            session.close()
            
            logger.info(f"Updated subscription {subscription_id}")
            return updated_subscription
        except Exception as e:
            logger.error(f"Error updating subscription {subscription_id}: {str(e)}")
            if session:
                session.rollback()
                session.close()
            raise
    
    # Signal operations
    def create_signal(self, signal_data: Dict[str, Any]) -> Signal:
        """
        Create a new trading signal.
        
        Args:
            signal_data (dict): Signal data
            
        Returns:
            Signal: Created signal object
        """
        try:
            session = self.Session()
            signal = Signal(**signal_data)
            session.add(signal)
            session.commit()
            signal_id = signal.id
            session.close()
            
            logger.info(f"Created signal with ID {signal_id}")
            return signal
        except Exception as e:
            logger.error(f"Error creating signal: {str(e)}")
            if session:
                session.rollback()
                session.close()
            raise
    
    def get_signals(self, symbol: Optional[str] = None, limit: int = 100) -> List[Signal]:
        """
        Get recent trading signals.
        
        Args:
            symbol (str, optional): Trading symbol to filter by
            limit (int): Maximum number of signals to retrieve
            
        Returns:
            list: List of signal objects
        """
        try:
            session = self.Session()
            query = session.query(Signal)
            
            if symbol:
                query = query.filter(Signal.symbol == symbol)
            
            signals = query.order_by(Signal.timestamp.desc()).limit(limit).all()
            session.close()
            
            return signals
        except Exception as e:
            logger.error(f"Error getting signals: {str(e)}")
            if session:
                session.close()
            raise
    
    def create_signal_outcome(self, outcome_data: Dict[str, Any]) -> SignalOutcome:
        """
        Create a new signal outcome.
        
        Args:
            outcome_data (dict): Outcome data
            
        Returns:
            SignalOutcome: Created signal outcome object
        """
        try:
            session = self.Session()
            outcome = SignalOutcome(**outcome_data)
            session.add(outcome)
            session.commit()
            outcome_id = outcome.id
            session.close()
            
            logger.info(f"Created signal outcome with ID {outcome_id}")
            return outcome
        except Exception as e:
            logger.error(f"Error creating signal outcome: {str(e)}")
            if session:
                session.rollback()
                session.close()
            raise
    
    # Market data operations
    def save_market_data(self, market_data: List[Dict[str, Any]]) -> List[MarketData]:
        """
        Save market data to the database.
        
        Args:
            market_data (list): List of market data dictionaries
            
        Returns:
            list: List of created MarketData objects
        """
        try:
            session = self.Session()
            market_data_objects = []
            
            for data in market_data:
                market_data_obj = MarketData(**data)
                session.add(market_data_obj)
                market_data_objects.append(market_data_obj)
            
            session.commit()
            session.close()
            
            logger.info(f"Saved {len(market_data_objects)} market data records")
            return market_data_objects
        except Exception as e:
            logger.error(f"Error saving market data: {str(e)}")
            if session:
                session.rollback()
                session.close()
            raise
    
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """
        Get market data for a specific symbol and date range.
        
        Args:
            symbol (str): Trading symbol
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of MarketData objects
        """
        try:
            session = self.Session()
            market_data = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timestamp >= start_date,
                MarketData.timestamp <= end_date
            ).order_by(MarketData.timestamp).all()
            session.close()
            
            return market_data
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            if session:
                session.close()
            raise


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the database manager
    db_manager = DBManager()
    
    # Create tables
    db_manager.create_tables()
    
    # Create a test user
    test_user = db_manager.create_user({
        "username": "testuser",
        "email": "test@example.com",
        "password_hash": "hashed_password"
    })
    
    # Create a test subscription
    test_subscription = db_manager.create_subscription({
        "user_id": test_user.id,
        "plan_name": "premium",
        "stripe_customer_id": "cus_123456",
        "stripe_subscription_id": "sub_123456",
        "start_date": datetime.utcnow(),
        "end_date": datetime.utcnow() + timedelta(days=30),
        "is_active": True
    })
    
    # Create a test signal
    test_signal = db_manager.create_signal({
        "symbol": "BTC/USDT",
        "signal_type": "BUY",
        "confidence": 0.85,
        "price": 30000.00,
        "volume": 100.00,
        "timestamp": datetime.utcnow(),
        "is_sent": False
    })
    
    # Create a test signal outcome
    test_outcome = db_manager.create_signal_outcome({
        "signal_id": test_signal.id,
        "exit_price": 32000.00,
        "exit_timestamp": datetime.utcnow() + timedelta(days=1),
        "return_pct": 6.67,
        "is_successful": True
    })
    
    # Save some test market data
    test_market_data = [
        {
            "symbol": "BTC/USDT",
            "timestamp": datetime.utcnow() - timedelta(hours=i),
            "open": 30000.00 - (i * 10),
            "high": 30100.00 - (i * 10),
            "low": 29900.00 - (i * 10),
            "close": 30050.00 - (i * 10),
            "volume": 100.00 + (i * 5)
        }
        for i in range(24)
    ]
    
    db_manager.save_market_data(test_market_data)
    
    # Test retrieving data
    user = db_manager.get_user(test_user.id)
    subscriptions = db_manager.get_user_subscriptions(test_user.id)
    signals = db_manager.get_signals(symbol="BTC/USDT")
    
    print(f"User: {user}")
    print(f"Subscriptions: {subscriptions}")
    print(f"Signals: {signals}")
