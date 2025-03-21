# ZombitX64 Discord Bot Commands Guide

This document provides a comprehensive guide for all commands available in the ZombitX64 Trading Signal Discord bot.

## Basic Commands

### `/start`
Starts the bot and shows welcome information.

**Example:**
```
/start
```

### `/help`
Shows a list of available commands and their descriptions.

**Example:**
```
/help
```

## User Account Commands

### `/connect <email>`
Connects your Discord account to your ZombitX64 platform account.

**Parameters:**
- `email` - Your registered email address on the ZombitX64 platform

**Example:**
```
/connect user@example.com
```

### `/status`
Shows your current subscription status, including plan, expiry date, and features.

**Example:**
```
/status
```

## Trading Signals

### `/signals`
Shows the latest trading signals available to you based on your subscription tier.

**Example:**
```
/signals
```

## Performance Reports

### `/report <period>`
Gets performance report for the specified period.

**Parameters:**
- `period` - Either "weekly" or "monthly"

**Examples:**
```
/report weekly
/report monthly
```

## Subscription Management

### `/redeem <code>`
Redeems a subscription code to activate or extend your subscription.

**Parameters:**
- `code` - The redeem code to use

**Example:**
```
/redeem ABC123XYZ
```

## VIP-only Commands

### `/market <symbol>`
Get detailed market analysis for a specific trading pair (VIP tier only).

**Parameters:**
- `symbol` - Trading pair (e.g., BTCUSDT, EURUSD)

**Example:**
```
/market BTCUSDT
```

### `/alert <symbol> <price>`
Set a price alert for a specific symbol (VIP tier only).

**Parameters:**
- `symbol` - Trading pair (e.g., BTCUSDT, EURUSD)
- `price` - Target price for the alert

**Example:**
```
/alert BTCUSDT 35000
```

## Admin Commands (Restricted)

These commands are only available to server administrators:

### `/broadcast <message>`
Broadcasts a message to all subscribed users.

### `/newcode <tier> <duration> <count>`
Generates new subscription redeem codes.

### `/stats`
Shows bot usage statistics and active subscriptions.

## Help and Support

If you need additional help or encounter issues with the bot, please contact our support team at support@zombitx64signals.com or join our [Support Server](https://discord.gg/zombitx64support).

## Command Permissions by Subscription Tier

| Command | Free | Basic | Premium | VIP |
|---------|:----:|:-----:|:-------:|:---:|
| `/start` | ✅ | ✅ | ✅ | ✅ |
| `/help` | ✅ | ✅ | ✅ | ✅ |
| `/connect` | ✅ | ✅ | ✅ | ✅ |
| `/status` | ✅ | ✅ | ✅ | ✅ |
| `/signals` | ✅* | ✅ | ✅ | ✅ |
| `/report weekly` | ❌ | ✅ | ✅ | ✅ |
| `/report monthly` | ❌ | ❌ | ✅ | ✅ |
| `/redeem` | ✅ | ✅ | ✅ | ✅ |
| `/market` | ❌ | ❌ | ✅ | ✅ |
| `/alert` | ❌ | ❌ | ❌ | ✅ |

*Free tier receives limited basic signals only
