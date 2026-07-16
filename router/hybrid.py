from fastapi import APIRouter
import logging

from config import config, AVAILABLE_MODELS

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('/models/available')
async def get_available_models():
    """Get list of available LLM models (all FREE)"""
    return {
        "default_provider": config.default_llm_provider.value,
        "default_model": config.default_llm_model,
        "available_models": {provider.value: models for provider, models in AVAILABLE_MODELS.items()},
        "usage_hint": "Send 'llm_provider' and 'llm_model' in your query request to switch models"
    }
