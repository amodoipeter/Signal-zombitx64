-- Enable required extensions
create extension if not exists "uuid-ossp";

-- Create enum types
create type subscription_tier as enum ('FREE', 'BASIC', 'PREMIUM', 'VIP');
create type subscription_status as enum ('ACTIVE', 'EXPIRED', 'CANCELED', 'TRIAL');
create type signal_type as enum ('BUY', 'SELL');
create type signal_status as enum ('ACTIVE', 'TP_HIT', 'SL_HIT', 'EXPIRED', 'CANCELED');

-- Create users table
create table if not exists users (
    id uuid primary key default uuid_generate_v4(),
    email text unique not null,
    username text unique not null,
    hashed_password text not null,
    is_active boolean default true,
    is_superuser boolean default false,
    telegram_id text,
    telegram_chat_id text,
    discord_id text,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- Create subscriptions table
create table if not exists subscriptions (
    id uuid primary key default uuid_generate_v4(),
    user_id uuid references users(id) on delete cascade,
    tier subscription_tier default 'FREE',
    status subscription_status default 'TRIAL',
    payment_method text,
    payment_id text,
    amount float,
    currency text default 'USD',
    start_date timestamptz not null,
    end_date timestamptz not null,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- Create signals table
create table if not exists signals (
    id uuid primary key default uuid_generate_v4(),
    symbol text not null,
    market text not null,
    signal_type signal_type not null,
    entry_price float not null,
    take_profit float not null,
    stop_loss float not null,
    risk_reward_ratio float not null,
    timeframe text not null,
    status signal_status default 'ACTIVE',
    confidence_score integer check (confidence_score >= 0 and confidence_score <= 100),
    analysis_summary text,
    chart_url text,
    entry_time timestamptz default now(),
    close_time timestamptz,
    profit_loss float,
    indicators_data jsonb,
    ai_model_version text not null,
    strategy_name text not null,
    strategy_category text not null,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- Create indexes
create index if not exists users_email_idx on users(email);
create index if not exists users_username_idx on users(username);
create index if not exists subscriptions_user_id_idx on subscriptions(user_id);
create index if not exists subscriptions_status_idx on subscriptions(status);
create index if not exists signals_symbol_idx on signals(symbol);
create index if not exists signals_market_idx on signals(market);
create index if not exists signals_signal_type_idx on signals(signal_type);
create index if not exists signals_status_idx on signals(status);
create index if not exists signals_strategy_name_idx on signals(strategy_name);
create index if not exists signals_strategy_category_idx on signals(strategy_category);

-- Create updated_at trigger function
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

-- Create triggers for updated_at
create trigger update_users_updated_at
    before update on users
    for each row
    execute function update_updated_at_column();

create trigger update_subscriptions_updated_at
    before update on subscriptions
    for each row
    execute function update_updated_at_column();

create trigger update_signals_updated_at
    before update on signals
    for each row
    execute function update_updated_at_column();