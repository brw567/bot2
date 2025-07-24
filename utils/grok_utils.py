"""Utility functions for interacting with the Grok API."""

import asyncio
import json
import logging
from typing import Any, Dict

import requests
import streamlit as st
from pydantic import BaseModel, ValidationError

from utils.telegram_utils import fetch_channel_messages
from config import GROK_API_KEY

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

class RiskResponse(BaseModel):
    """
    Pydantic model for Grok risk assessment response.
    Ensures strict JSON output for trading decisions.
    """
    trade: str
    sl_mult: float
    tp_mult: float
    details: str  # Added for reasoning

def grok_api_call(prompt: Dict[str, Any]) -> BaseModel:
    """Call Grok with a structured prompt and validate the JSON response."""
    try:
        headers = {
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json",
        }
        # Instruct Grok to reply with JSON only so that we can parse reliably
        prompt["instructions"] = (
            "Output ONLY valid JSON matching the provided schema. "
            "Include details explaining reasoning."
        )
        data = {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": json.dumps(prompt)}],
        }

        logging.debug("Sending Grok prompt: %s", prompt)
        response = requests.post(GROK_API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = json.loads(response.json()["choices"][0]["message"]["content"])
        
        logging.debug("Grok raw response: %s", result)

        # Validate based on requested task
        if prompt.get("task") == "sentiment_analysis":
            validated = SentimentResponse(**result)
            logging.info("Grok sentiment response: %s", validated.json())
            return validated
        if prompt.get("task") == "risk_assessment":
            validated = RiskResponse(**result)
            logging.info("Grok risk response: %s", validated.json())
            return validated
        raise ValueError(f"Unknown task: {prompt.get('task')}")
    except (requests.RequestException, json.JSONDecodeError, ValidationError, ValueError) as e:
        logging.error("Grok API call failed: %s", e)
        if prompt.get("task") == "sentiment_analysis":
            return SentimentResponse(sentiment="neutral", score=0.0, details="Validation failed")
        return RiskResponse(trade="no", sl_mult=0.0, tp_mult=0.0, details="Validation failed")

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
            channel_msgs = await fetch_channel_messages(channel, limit=100)
            logging.debug("Fetched %d messages from %s", len(channel_msgs), channel)
            messages.extend(channel_msgs)
        prompt = {
            "task": "sentiment_analysis",
            "symbol": symbol,
            "messages": ' '.join(messages[:20]),  # Limit to 20 to avoid token limits
            "output_schema": {"sentiment": "positive/negative/neutral", "score": "float", "details": "str"}
        }
        logging.debug("Aggregated %d messages for sentiment analysis", len(messages))
        return grok_api_call(prompt)
    except Exception as e:
        logging.error(f"Sentiment analysis failed for {symbol}: {e}")
        return SentimentResponse(sentiment="neutral", score=0.0, details="Error in processing")

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
                "details": "str",
            },
        }
        logging.debug("Requesting risk assessment: %s", prompt)
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
        logging.debug("Requesting parameter recommendation: %s", prompt)
        result = grok_api_call(prompt)
        if isinstance(result, dict) and "value" in result:
            return float(result["value"])
        if hasattr(result, "value"):
            return float(result.value)
        return 0.0
    except Exception as e:
        logging.error("Parameter recommendation failed for %s %s: %s", symbol, param, e)
        return 0.0
