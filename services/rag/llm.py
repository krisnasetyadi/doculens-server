"""Choosing, loading and caching LLMs (Gemini, HuggingFace) and the embeddings.

Mixin of services.rag.processor.PDFQAProcessor; the methods were moved
verbatim from that class by MS-657."""
import logging
import torch
from config import AVAILABLE_MODELS, LLMProvider, config
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, pipeline
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMMixin:
    def get_llm(self, provider: Optional[str] = None, model: Optional[str] = None) -> Tuple[Any, str]:
        """
        Get LLM instance based on provider and model.
        Returns (llm_instance, model_identifier_string)
        
        If provider/model not specified, uses config defaults.
        Caches LLM instances for reuse.
        """
        # Determine provider/model defaults for request without explicit override
        if not provider and not model:
            llm_provider = config.default_llm_provider
            llm_model = config.default_llm_model
        elif provider:
            try:
                llm_provider = LLMProvider(provider.lower())
            except ValueError:
                logger.warning(f"Invalid provider '{provider}', using default")
                llm_provider = config.default_llm_provider
            llm_model = model or (config.gemini_model if llm_provider == LLMProvider.GEMINI else config.model_name)
        else:
            inferred_gemini = bool(model and model.lower().startswith("gemini"))
            llm_provider = LLMProvider.GEMINI if inferred_gemini else config.default_llm_provider
            llm_model = model or config.default_llm_model
        
        # Create cache key
        cache_key = f"{llm_provider.value}:{llm_model}"
        model_identifier = f"{llm_provider.value}/{llm_model}"
        
        # Check cache
        if cache_key in self._llm_cache:
            logger.info(f"🔄 Using cached LLM: {model_identifier}")
            return self._llm_cache[cache_key], model_identifier
        
        # Load new LLM
        logger.info(f"🚀 Loading LLM: {model_identifier}")
        
        try:
            if llm_provider == LLMProvider.HUGGINGFACE:
                llm = self._load_huggingface_llm(llm_model)
            elif llm_provider == LLMProvider.GEMINI:
                llm = self._load_gemini_llm(llm_model)
            else:
                raise ValueError(f"Unsupported provider: {llm_provider}")
            
            # Cache the LLM
            self._llm_cache[cache_key] = llm
            self._current_provider = llm_provider
            self._current_model = llm_model
            
            logger.info(f"✅ LLM loaded successfully: {model_identifier}")
            return llm, model_identifier
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_identifier}: {e}")
            if provider or model:
                logger.info("⚠️ Falling back to configured default LLM")
                return self.get_llm(config.default_llm_provider.value, config.default_llm_model)
            raise

    def _load_huggingface_llm(self, model_name: str):
        """Load HuggingFace model (local)"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            logger.warning(f"Model {model_name} failed, using fallback")
            model_name = "google/flan-t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            batch_size=4 if torch.cuda.is_available() else 1
        )
        
        return HuggingFacePipeline(pipeline=pipe)

    def _load_gemini_llm(self, model_name: str):
        """Load Google Gemini model (cloud - free tier)"""
        if not config.gemini_api_key:
            logger.error("❌ GEMINI_API_KEY not set in .env file")
            raise ValueError("GEMINI_API_KEY not configured. Please add it to .env file or use HuggingFace/Ollama")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.error("❌ langchain-google-genai not installed")
            raise ImportError("Please install: pip install langchain-google-genai")
        
        logger.info(f"🔑 Using Gemini API with model: {model_name}")
        
        # Fix model name - Gemini uses different naming convention
        # Map common names to actual Gemini model names
        model_mapping = {
            'gemini-1.5-flash': 'gemini-1.5-flash-latest',
            'gemini-1.5-pro': 'gemini-1.5-pro-latest',
            'gemini-pro': 'gemini-1.5-pro-latest',
        }
        
        actual_model = model_mapping.get(model_name, model_name)
        if actual_model != model_name:
            logger.info(f"🔄 Model name mapped: {model_name} -> {actual_model}")
        
        # See config.gemini_thinking_budget for why thinking is off by default.
        extra_kwargs: Dict[str, Any] = {}
        if config.gemini_thinking_budget is not None and "flash" in actual_model.lower():
            extra_kwargs["thinking_budget"] = config.gemini_thinking_budget

        return ChatGoogleGenerativeAI(
            model=actual_model,
            google_api_key=config.gemini_api_key,
            temperature=config.temperature,
            **extra_kwargs,
        )

    def get_available_models(self) -> Dict[str, List[str]]:
        """Return available models per provider"""
        return {provider.value: models for provider, models in AVAILABLE_MODELS.items()}

    def get_current_model_info(self) -> str:
        """Get current model identifier string"""
        if self._current_provider and self._current_model:
            return f"{self._current_provider.value}/{self._current_model}"
        return f"{config.default_llm_provider.value}/{config.default_llm_model}"

    def initialize_components(self):
        """Initialize all components including database"""
        with self._init_lock:
            if self._initialized:
                return

            logger.info("Initializing NLP components...")
            try:
                # Initialize embeddings first (faster to load)
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.embedding_model,
                    model_kwargs={
                        'device': 'cuda' if torch.cuda.is_available()
                        else 'cpu'
                    },
                    encode_kwargs={'normalize_embeddings': True}
                )

                # Load default LLM
                self.llm, model_id = self.get_llm()
                
                # Keep tokenizer reference for HuggingFace models
                if config.default_llm_provider == LLMProvider.HUGGINGFACE:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                    except:
                        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

                self.initialize_database()
            
                self._initialized = True
                logger.info(f"✅ All components initialized with model: {model_id}")

            except Exception as e:
                logger.error(f"Failed to initialize components: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Failed to initialize components: {str(e)}")
                raise
