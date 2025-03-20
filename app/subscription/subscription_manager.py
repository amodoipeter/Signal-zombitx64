"""
Subscription & Billing module for the AI Signal Provider.
This module handles user subscriptions and payments via Stripe.
"""

import os
import logging
import stripe
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
stripe.api_key = os.getenv("STRIPE_API_KEY")
default_plan_id = os.getenv("DEFAULT_SUBSCRIPTION_PLAN_ID")

class SubscriptionManager:
    """
    A class to manage user subscriptions and payment processing.
    Handles subscription creation, cancellation, and renewal.
    """
    
    def __init__(self):
        """
        Initialize the SubscriptionManager.
        """
        if not stripe.api_key:
            raise ValueError("Stripe API key not found in environment variables")
        
        # Define subscription plans
        self.plans = {
            "basic": {
                "name": "Basic",
                "price_id": os.getenv("BASIC_PLAN_PRICE_ID"),
                "features": ["Daily signals", "Web dashboard access"],
                "amount": 29.99,
                "currency": "USD",
                "interval": "month"
            },
            "premium": {
                "name": "Premium",
                "price_id": os.getenv("PREMIUM_PLAN_PRICE_ID"),
                "features": ["Real-time signals", "Web dashboard access", "Priority support"],
                "amount": 59.99,
                "currency": "USD",
                "interval": "month"
            },
            "pro": {
                "name": "Professional",
                "price_id": os.getenv("PRO_PLAN_PRICE_ID"),
                "features": ["Real-time signals", "Web dashboard access", "Priority support", "Custom alerts"],
                "amount": 99.99,
                "currency": "USD",
                "interval": "month"
            }
        }
        
        logger.info("Subscription manager initialized")
    
    def get_available_plans(self):
        """
        Get all available subscription plans.
        
        Returns:
            dict: Available subscription plans
        """
        return self.plans
    
    def create_customer(self, user_data):
        """
        Create a new customer in Stripe.
        
        Args:
            user_data (dict): User information including email, name, etc.
            
        Returns:
            dict: Stripe customer object
        """
        try:
            customer = stripe.Customer.create(
                email=user_data.get("email"),
                name=user_data.get("name"),
                metadata={
                    "user_id": str(user_data.get("user_id")),
                    "registration_date": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Created Stripe customer: {customer.id}")
            return customer
        except stripe.error.StripeError as e:
            logger.error(f"Error creating Stripe customer: {str(e)}")
            raise
    
    def create_subscription(self, customer_id, plan_id=None):
        """
        Create a new subscription for a customer.
        
        Args:
            customer_id (str): Stripe customer ID
            plan_id (str, optional): ID of the plan to subscribe to
            
        Returns:
            dict: Stripe subscription object
        """
        # Use default plan if none specified
        if plan_id is None:
            plan_id = default_plan_id
            
        if not plan_id:
            raise ValueError("No plan ID specified and no default plan available")
        
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": plan_id}],
                expand=["latest_invoice.payment_intent"]
            )
            
            logger.info(f"Created subscription {subscription.id} for customer {customer_id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error creating subscription: {str(e)}")
            raise
    
    def cancel_subscription(self, subscription_id):
        """
        Cancel a subscription.
        
        Args:
            subscription_id (str): Stripe subscription ID
            
        Returns:
            dict: Stripe subscription object
        """
        try:
            subscription = stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
            
            logger.info(f"Subscription {subscription_id} set to cancel at period end")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error canceling subscription: {str(e)}")
            raise
    
    def immediate_cancel_subscription(self, subscription_id):
        """
        Cancel a subscription immediately.
        
        Args:
            subscription_id (str): Stripe subscription ID
            
        Returns:
            dict: Stripe subscription object
        """
        try:
            subscription = stripe.Subscription.delete(subscription_id)
            
            logger.info(f"Subscription {subscription_id} cancelled immediately")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error canceling subscription immediately: {str(e)}")
            raise
    
    def update_subscription(self, subscription_id, new_plan_id):
        """
        Update a subscription to a new plan.
        
        Args:
            subscription_id (str): Stripe subscription ID
            new_plan_id (str): ID of the new plan
            
        Returns:
            dict: Stripe subscription object
        """
        try:
            # Get the subscription
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            # Update the subscription item
            subscription = stripe.Subscription.modify(
                subscription_id,
                items=[{
                    "id": subscription["items"]["data"][0].id,
                    "price": new_plan_id
                }]
            )
            
            logger.info(f"Updated subscription {subscription_id} to plan {new_plan_id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error updating subscription: {str(e)}")
            raise
    
    def get_subscription(self, subscription_id):
        """
        Get details of a subscription.
        
        Args:
            subscription_id (str): Stripe subscription ID
            
        Returns:
            dict: Stripe subscription object
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            logger.info(f"Retrieved subscription {subscription_id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Error retrieving subscription: {str(e)}")
            raise
    
    def get_customer_subscriptions(self, customer_id):
        """
        Get all subscriptions for a customer.
        
        Args:
            customer_id (str): Stripe customer ID
            
        Returns:
            list: List of Stripe subscription objects
        """
        try:
            subscriptions = stripe.Subscription.list(customer=customer_id)
            
            logger.info(f"Retrieved {len(subscriptions.data)} subscriptions for customer {customer_id}")
            return subscriptions.data
        except stripe.error.StripeError as e:
            logger.error(f"Error retrieving customer subscriptions: {str(e)}")
            raise
    
    def create_checkout_session(self, customer_id, plan_id, success_url, cancel_url):
        """
        Create a checkout session for a customer to subscribe to a plan.
        
        Args:
            customer_id (str): Stripe customer ID
            plan_id (str): ID of the plan to subscribe to
            success_url (str): URL to redirect to on successful payment
            cancel_url (str): URL to redirect to if payment is cancelled
            
        Returns:
            dict: Stripe checkout session object
        """
        try:
            session = stripe.checkout.Session.create(
                customer=customer_id,
                payment_method_types=["card"],
                line_items=[{
                    "price": plan_id,
                    "quantity": 1
                }],
                mode="subscription",
                success_url=success_url,
                cancel_url=cancel_url
            )
            
            logger.info(f"Created checkout session {session.id} for customer {customer_id}")
            return session
        except stripe.error.StripeError as e:
            logger.error(f"Error creating checkout session: {str(e)}")
            raise
    
    def handle_webhook(self, payload, sig_header):
        """
        Handle Stripe webhook events.
        
        Args:
            payload (str): Webhook payload
            sig_header (str): Signature header from Stripe
            
        Returns:
            dict: Event data
        """
        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        if not webhook_secret:
            raise ValueError("Stripe webhook secret not found in environment variables")
        
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
            
            # Handle specific event types
            if event.type == "checkout.session.completed":
                # Payment was successful, provision access
                session = event.data.object
                customer_id = session.customer
                subscription_id = session.subscription
                
                logger.info(f"Checkout completed for customer {customer_id}, subscription {subscription_id}")
                
                # Here you would update your database to record the subscription
                
            elif event.type == "customer.subscription.updated":
                subscription = event.data.object
                customer_id = subscription.customer
                
                logger.info(f"Subscription {subscription.id} updated for customer {customer_id}")
                
                # Here you would update your database with the subscription changes
                
            elif event.type == "customer.subscription.deleted":
                subscription = event.data.object
                customer_id = subscription.customer
                
                logger.info(f"Subscription {subscription.id} deleted for customer {customer_id}")
                
                # Here you would update your database to revoke access
            
            return event.data.object
        except (ValueError, stripe.error.SignatureVerificationError) as e:
            logger.error(f"Error verifying webhook signature: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the subscription manager
    subscription_manager = SubscriptionManager()
    
    # Get available plans
    plans = subscription_manager.get_available_plans()
    print(f"Available plans: {plans}")
    
    # Create a test customer and subscription
    # Note: This is just for demonstration and would require actual Stripe API keys to work
    try:
        # Create a customer
        customer = subscription_manager.create_customer({
            "email": "test@example.com",
            "name": "Test User",
            "user_id": "123456"
        })
        
        # Create a subscription
        subscription = subscription_manager.create_subscription(
            customer.id,
            plans["basic"]["price_id"]
        )
        
        print(f"Created subscription: {subscription.id}")
    except Exception as e:
        print(f"Error in example: {str(e)}")
