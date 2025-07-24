import requests
import json
import logging
import streamlit as st
from pydantic import BaseModel, ValidationError
from utils.telegram_utils import fetch_channel_messages
from config import GROK_API_KEY, GROK_TIMEOUT

logging.basicConfig(level=logging.INFO, filename='bot.log', filemode='a', format='%(asctime)s - %(message)s')

GROK_API_URL = "https://api.x.ai/v1/chat/completions"  # Grok API endpoint (per xAI docs, July 2025)

class SentimentResponse(BaseModel):
    """
    Pydantic model for Grok sentiment analysis response.
    Ensures strict JSON output validation.
    """
    sentiment: str
    score: float
    details: str  # Added for trustworthiness, per immediate task

class MultiSentimentResponse(BaseModel):
    """Mapping of symbols to their sentiment response."""
    __root__: dict[str, SentimentResponse]

class RiskResponse(BaseModel):
    """
    Pydantic model for Grok risk assessment response.
    Ensures strict JSON output for trading decisions.
    """
    trade: str
    sl_mult: float
    tp_mult: float
    details: str  # Added for reasoning

def grok_api_call(prompt):
    """
    Make a Grok API call with structured JSON prompt.

    Args:
        prompt (dict): Structured JSON prompt with task, data, and output schema.

    Returns:
        BaseModel: Validated Pydantic model (SentimentResponse or RiskResponse).

    Note: Uses Pydantic for strict output validation, addressing immediate task.
    Instructs Grok to return only JSON matching schema for reliability.
    """
    try:
        headers = {'Authorization': f'Bearer {GROK_API_KEY}', 'Content-Type': 'application/json'}
        prompt['instructions'] = "Output ONLY valid JSON matching the provided schema. Include details explaining reasoning."
        data = {'model': 'grok-beta', 'messages': [{'role': 'user', 'content': json.dumps(prompt)}]}
        response = requests.post(GROK_API_URL, headers=headers, json=data, timeout=st.session_state.get("grok_timeout", GROK_TIMEOUT))
        logging.info(f"Grok task {prompt.get('task')}: {response.text}")
        response.raise_for_status()
        result = json.loads(response.json()['choices'][0]['message']['content'])
        
        # Validate based on task
        if prompt.get('task') == 'sentiment_analysis':
            validated = SentimentResponse(**result)
            logging.info(f"{prompt.get('task')} succeeded")
            return validated
        elif prompt.get('task') == 'multi_sentiment':
            validated = MultiSentimentResponse(__root__=
                                               {k: SentimentResponse(**v) for k, v in result.items()})
            logging.info(f"{prompt.get('task')} succeeded")
            return validated.__root__
        elif prompt.get('task') == 'risk_assessment':
            validated = RiskResponse(**result)
            logging.info(f"{prompt.get('task')} succeeded")
            return validated
        else:
            raise ValueError(f"Unknown task: {prompt.get('task')}")
    except (requests.RequestException, json.JSONDecodeError, ValidationError) as e:
        logging.error(f"Grok API call failed: {e}")
        if prompt.get('task') == 'multi_sentiment':
            return {}
        return SentimentResponse(sentiment="neutral", score=0.0, details="Validation failed") if prompt.get('task') == 'sentiment_analysis' else RiskResponse(trade="no", sl_mult=0.0, tp_mult=0.0, details="Validation failed")

async def get_sentiment_analysis(symbol):
    """
    Fetch Telegram channel messages and analyze sentiment via Grok.

    Args:
        symbol (str): Asset symbol (e.g., 'BTC').

    Returns:
        SentimentResponse: Pydantic model with sentiment, score, and details.

    Note: Uses fetch_channel_messages (up to 100 messages with timestamp filtering)
    per immediate task; limits messages to avoid token overflow.
    """
    try:
        channels = st.session_state.get('channels', ['crypto_news_channel'])
        messages = []
        for channel in channels:
            channel_msgs = await fetch_channel_messages(channel, limit=100)  # Extended limit, timestamp filtered
            messages.extend(channel_msgs)
        logging.info(f"{len(messages)} messages gathered for sentiment analysis")
        prompt = {
            "task": "sentiment_analysis",
            "symbol": symbol,
            "messages": ' '.join(messages[:20]),  # Limit to 20 to avoid token limits
            "output_schema": {"sentiment": "positive/negative/neutral", "score": "float", "details": "str"}
        }
        return grok_api_call(prompt)
    except Exception as e:
        logging.error(f"Sentiment analysis failed for {symbol}: {e}")
        return SentimentResponse(sentiment="neutral", score=0.0, details="Error in processing")


async def get_multi_sentiment_analysis(symbols):
    """Fetch sentiment analysis for multiple symbols in a single Grok call."""
    try:
        channels = st.session_state.get('channels', ['crypto_news_channel'])
        messages = []
        for channel in channels:
            channel_msgs = await fetch_channel_messages(channel, limit=100)
            messages.extend(channel_msgs)
        logging.info(f"{len(messages)} messages gathered for sentiment analysis")
        prompt = {
            "task": "multi_sentiment",
            "symbols": symbols,
            "messages": ' '.join(messages[:20]),
            "output_schema": {
                "symbol": "str",
                "sentiment": "positive/negative/neutral",
                "score": "float",
                "details": "str"
            }
        }
        result = grok_api_call(prompt)
        if not isinstance(result, dict):
            raise ValueError("Invalid response format")
        missing = [s for s in symbols if s not in result]
        if missing:
            raise ValueError(f"Missing sentiment data for {', '.join(missing)}")
        return result
    except Exception as e:
        logging.error(f"Multi-sentiment analysis failed: {e}")
        return {sym: SentimentResponse(sentiment='neutral', score=0.0, details='Error') for sym in symbols}

def get_risk_assessment(symbol, price, vol, winrate):
    """
    Fetch risk assessment from Grok for trading decisions.

    Args:
        symbol (str): Trading pair (e.g., 'BTC/USDT').
        price (float): Current price.
        vol (float): Volatility (e.g., (high-low)/low).
        winrate (float): Current winrate.

    Returns:
        RiskResponse: Pydantic model with trade decision, SL/TP multipliers, details.

    Note: Structured prompt ensures consistent output, per immediate task.
    """
    try:
        prompt = {
            "task": "risk_assessment",
            "symbol": symbol,
            "data": {"current_price": price, "volatility": vol, "win_rate": winrate},
            "output_schema": {
                "trade": "yes/no",
                "sl_mult": "float",
                "tp_mult": "float",
                "details": "str"
            }
        }
        return grok_api_call(prompt)
    except Exception as e:
        logging.error(f"Risk assessment failed for {symbol}: {e}")
        return RiskResponse(trade="no", sl_mult=0.0, tp_mult=0.0, details="Error in processing")


def get_grok_recommendation(symbol: str, param: str) -> float:
    """Request a recommended parameter value from Grok."""
    try:
        prompt = {
            "task": "parameter_tuning",
            "symbol": symbol,
            "parameter": param,
            "output_schema": {"value": "float"},
        }
        result = grok_api_call(prompt)
        if isinstance(result, dict) and "value" in result:
            return float(result["value"])
        if hasattr(result, "value"):
            return float(result.value)
        return 0.0
    except Exception as e:
        logging.error(f"Parameter recommendation failed for {symbol} {param}: {e}")
        return 0.0
