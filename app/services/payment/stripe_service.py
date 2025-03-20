import stripe
import logging
from typing import Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class StripeService:
    def __init__(self):
        """Initialize Stripe with API key."""
        stripe.api_key = settings.STRIPE_API_KEY
        self.webhook_secret = settings.STRIPE_WEBHOOK_SECRET
    
    async def create_checkout_session(
        self, 
        customer_email: str,
        price_amount: int,  # in cents
        product_name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Create a Stripe checkout session for subscription payment.
        
        Args:
            customer_email: Customer's email
            price_amount: Price in cents (e.g., 2999 for $29.99)
            product_name: Name of the product/subscription
            metadata: Additional metadata for the session
            
        Returns:
            Stripe checkout session
        """
        try:
            checkout_session = stripe.checkout.Session.create(
                customer_email=customer_email,
                payment_method_types=['card'],
                line_items=[
                    {
                        'price_data': {
                            'currency': 'usd',
                            'product_data': {
                                'name': product_name,
                                'description': 'AI Trading Signals Subscription',
                            },
                            'unit_amount': price_amount,
                        },
                        'quantity': 1,
                    },
                ],
                mode='payment',
                success_url=f"{settings.FRONTEND_URL}/payment/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{settings.FRONTEND_URL}/payment/cancel",
                metadata=metadata,
            )
            
            return checkout_session
        except Exception as e:
            logger.error(f"Error creating Stripe checkout session: {str(e)}")
            raise
    
    async def construct_event(self, payload, sig_header):
        """Construct a Stripe event from webhook."""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            return event
        except ValueError as e:
            # Invalid payload
            logger.error(f"Invalid Stripe payload: {str(e)}")
            raise
        except stripe.error.SignatureVerificationError as e:
            # Invalid signature
            logger.error(f"Invalid Stripe signature: {str(e)}")
            raise
        
    async def retrieve_session(self, session_id: str):
        """Retrieve a checkout session by ID."""
        try:
            return stripe.checkout.Session.retrieve(session_id)
        except Exception as e:
            logger.error(f"Error retrieving Stripe session: {str(e)}")
            raise

# Create the service instance
stripe_service = StripeService()
