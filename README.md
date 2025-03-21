# ZombitX64 AI Signal Provider

An AI-powered trading signal provider system with Telegram integration and subscription model.

## 🧠 Features

- **AI Signal Generation**: Uses machine learning models to analyze market data and generate trading signals
- **Telegram Bot Integration**: Sends signals directly to users via Telegram
- **Subscription Model**: Monetization through Stripe payment processing or redeem codes
- **Real-time Updates**: Notifies users when TP or SL are hit
- **Admin Dashboard**: Manage signals, users, and subscriptions
- **Performance Analytics**: Weekly and monthly win rate reports sent to subscribers
- **Redeem Codes**: Generate subscription codes with predefined durations (7, 15, 30 days or forever)
- **Auto-Expiration**: Automatically removes users from premium groups when subscriptions expire

## 🔄 System Architecture

```mermaid
graph TD
    A[Market Data API] -->|Fetches price data| B[Data Processor]
    B -->|Processed data| C[AI Signal Generator]
    C -->|Creates signals| D[Signal Database]
    D -->|Retrieves signals| E[Notification Service]
    E -->|Sends signals| F[Telegram Bot]
    H[Users] -->|Subscribe| I[Payment Service]
    I -->|Payment confirmation| J[Subscription DB]
    J -->|Access control| E
    D -->|Analyzes| K[Performance Analytics]
    K -->|Reports| E
```

## 🚀 Signal Generation Workflow

```mermaid
flowchart LR
    A[Market Data] -->|Real-time prices| B{Technical Analysis}
    B -->|Indicators| C[AI Model]
    B -->|Chart Patterns| C
    C -->|Prediction| D{Confidence Check}
    D -->|Low Confidence| E[Discard]
    D -->|High Confidence| F[Generate Signal]
    F -->|Save| G[Database]
    F -->|Notify| H[Send to Subscribers]
    G -->|Monitor| I{Track Progress}
    I -->|TP Hit| J[Success Update]
    I -->|SL Hit| K[Loss Update]
    I -->|Expired| L[Expiration Update]
    J & K & L -->|Analytics| M[Performance Reports]
```

## 📈 Performance Analytics Mind Map

```mermaid
mindmap
  root((Analytics))
    Win Rate Calculation
      Weekly Reports
      Monthly Reports
      By Market Type
    Signal Performance
      Success Rate
      Average Profit
      Average Loss
      Risk-Reward Ratio
    Market Analysis
      Best Performing Pairs
      Best Timeframes
      Market Correlation
    User Metrics
      Active Subscribers
      Subscription Tiers
      Renewal Rate
```

## 💳 Subscription System Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Stripe
    participant Database
    participant Bot

    User->>API: Request subscription
    API->>Stripe: Create checkout session
    Stripe-->>API: Return checkout URL
    API-->>User: Redirect to payment
    User->>Stripe: Complete payment
    Stripe->>API: Webhook notification
    API->>Database: Create/Update subscription
    API->>Bot: Notify subscription active
    Bot-->>User: Welcome message
```

## 📱 Telegram Bot Integration

```mermaid
flowchart TD
    A[Signal Generated] -->|New Signal| B{Is User Subscribed?}
    B -->|Yes| C[Format Signal Message]
    B -->|No| D[Ignore or Send Teaser]
    C -->|Generate Chart Image| E[Prepare Media]
    C -->|Format Text| F[Prepare Message]
    E & F -->|Combined| G[Send to User]
    H[Signal Update] -->|TP/SL Hit| I[Send Update Notification]
    J[Schedule] -->|Weekly Report| K[Generate Win Rate Report]
    J -->|Monthly Report| K
    K -->|Format Report| L[Send to Subscribers]
```

## 🚀 Tech Stack

- **Backend**: Python + FastAPI
- **Database**: PostgreSQL
- **AI Models**: NumPy, Pandas, TensorFlow/PyTorch
- **Market Data**: Integration with Binance API
- **Bots**: Telegram Bot API (aiogram)
- **Payments**: Stripe API
- **Deployment**: Docker, Kubernetes

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL
- Binance API Key
- Telegram Bot Token
- Stripe API Key

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Signal-zombitx64.git
   cd Signal-zombitx64
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

3. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Alternatively, install dependencies and run locally**
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

## 📝 API Documentation

After starting the application, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📊 Signal Generation System

The system uses a combination of technical indicators and AI models to generate trading signals:

```
Signal Generation Logic
│
├── Technical Analysis
│   ├── RSI (Relative Strength Index)
│   ├── MACD (Moving Average Convergence Divergence)
│   ├── Bollinger Bands
│   └── Ichimoku Cloud
│
├── Machine Learning
│   ├── Feature Extraction
│   ├── Signal Classification
│   └── Confidence Scoring
│
└── Signal Validation
    ├── Risk/Reward Calculation
    ├── Market Condition Check
    └── Strategy Selection
```

## 💰 Subscription Tiers

- **FREE**: Basic signals with limited features
- **BASIC**: $29.99/month - All crypto signals
- **PREMIUM**: $49.99/month - Crypto + forex signals
- **VIP**: $99.99/month - All signals + exclusive strategies

## 🎟️ Redeem Code System

The platform offers a flexible redeem code system:

```mermaid
flowchart TD
    A[Admin] -->|Creates| B[Redeem Code]
    B -->|Has Duration| C{Duration Type}
    C -->|7 Days| D[Week Access]
    C -->|15 Days| E[Bi-weekly Access]
    C -->|30 Days| F[Monthly Access]
    C -->|Forever| G[Lifetime Access]
    C -->|Custom| H[Custom Duration]
    I[User] -->|Redeems| B
    B -->|Success| J[Subscription Created]
    J -->|On Expiration| K[Auto Removal from Groups]
    K -->|Notification| I
    J -->|Premium Access| L[Access Premium Services]
```

### Available Durations:
- **7-Day Access**: Perfect for trial subscriptions
- **15-Day Access**: Bi-weekly subscription
- **30-Day Access**: Standard monthly subscription
- **Lifetime Access**: Permanent subscription without expiration
- **Custom Duration**: Admin-defined custom subscription length

## 📱 Bot Commands

### Telegram Bot
- `/start` - Start the bot and register
- `/help` - Show help information
- `/signals` - Get latest signals
- `/subscription` - Check subscription status
- `/weekly_report` - Request weekly performance report
- `/monthly_report` - Request monthly performance report
- `/redeem` - Redeem a subscription code

## 🧪 Testing

```bash
pytest
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.