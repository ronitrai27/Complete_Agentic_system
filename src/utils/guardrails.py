import re
import os
from typing import Tuple, Optional
from loguru import logger

# NeMo API key (loaded from environment)
NEMO_API_KEY = None

# Predefined greeting check dictionary
GREETINGS_MAP = {
    "hi": "Hello! How can I help you today with your tech docs or web search?",
    "hello": "Hello! How can I help you today with your tech docs or web search?",
    "hy": "Hello! How can I help you today with your tech docs or web search?",
    "hey": "Hello! How can I help you today with your tech docs or web search?",
    "hey there": "Hello! How can I help you today with your tech docs or web search?",
    "hello there": "Hello! How can I help you today with your tech docs or web search?",
    "hi there": "Hello! How can I help you today with your tech docs or web search?",
    "hy there": "Hello! How can I help you today with your tech docs or web search?",
    "hello bot": "Hello! How can I help you today with your tech docs or web search?",
    "hey bot": "Hello! How can I help you today with your tech docs or web search?",
    "hi bot": "Hello! How can I help you today with your tech docs or web search?",
    "hello assistant": "Hello! How can I help you today with your tech docs or web search?",
    "hi assistant": "Hello! How can I help you today with your tech docs or web search?",
    "good morning": "Hello! How can I help you today with your tech docs or web search?",
    "good afternoon": "Hello! How can I help you today with your tech docs or web search?",
    "good evening": "Hello! How can I help you today with your tech docs or web search?",
    "bye": "Goodbye! Let me know if you need any other tech assistance.",
    "goodbye": "Goodbye! Let me know if you need any other tech assistance.",
    "see ya": "Goodbye! Let me know if you need any other tech assistance.",
    "thank you": "You're welcome! Let me know if you have other tech questions.",
    "thanks": "You're welcome! Let me know if you have other tech questions.",
    "thanks a lot": "You're welcome! Let me know if you have other tech questions.",
    "thank you so much": "You're welcome! Let me know if you have other tech questions.",
    "welcome": "You're welcome! Let me know if you have other tech questions.",
    "you're welcome": "You're welcome! Let me know if you have other tech questions.",
}

def check_fast_path_greeting(query: str) -> Optional[str]:
    """
    Checks if a query is a greeting or standard polite phrase.
    Returns a predefined response string if matched, else None.
    Guarantees zero LLM latency/cost for these interactions.
    """
    from src.utils.event_bus import emit
    clean_query = query.strip().lower()
    # Remove standard trailing punctuation for matching
    clean_query = re.sub(r"[^\w\s]", "", clean_query).strip()
    
    msg_start = f"🔍 [Guardrails] Pre-checking query for fast-path greetings: '{query}'"
    print(msg_start, flush=True)
    emit(msg_start, "step")
    
    if clean_query in GREETINGS_MAP:
        resp = GREETINGS_MAP[clean_query]
        msg_match = f"ℹ️ [Guardrails] Fast-path greeting match found: '{query}' -> Predefined response: '{resp}' (Bypassed LLM completely)"
        print(msg_match, flush=True)
        emit(msg_match, "success")
        return resp
        
    msg_no_match = f"✅ [Guardrails] No fast-path greeting match found. Query proceeds to safety guardrails check."
    print(msg_no_match, flush=True)
    emit(msg_no_match, "step")
    return None


_RAILS_INSTANCE = None

def get_rails_instance():
    """
    Returns a cached instance of NeMo Guardrails LLMRails.
    """
    global _RAILS_INSTANCE
    if _RAILS_INSTANCE is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("OPENAI_API_KEY")
        orig_key = os.environ.get("OPENAI_API_KEY")
        try:
            # Set key temporarily only during initialization
            if key:
                os.environ["OPENAI_API_KEY"] = key
            from nemoguardrails import LLMRails, RailsConfig
            # NeMo Guardrails needs nest_asyncio in environments with active loops
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except Exception:
                pass

            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(current_dir), "guardrails")
            logger.info(f"[Guardrails] Loading NeMo config from path: {config_path}")
            
            config = RailsConfig.from_path(config_path)
            _RAILS_INSTANCE = LLMRails(config)
            logger.info("[Guardrails] NeMo Guardrails successfully initialized.")
        except Exception as e:
            logger.error(f"[Guardrails] Failed to initialize NeMo Guardrails: {e}")
            raise e
        finally:
            # Restore original key
            if orig_key is not None:
                os.environ["OPENAI_API_KEY"] = orig_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            
    return _RAILS_INSTANCE


def check_input_guardrails(query: str) -> Tuple[bool, Optional[str]]:
    """
    Runs NeMo Guardrails against user query.
    Returns:
        (is_blocked, refusal_message)
    """
    from src.utils.event_bus import emit
    
    msg_start = f"🛡️ [Guardrails] Running NeMo Guardrails safety checks for query: '{query}'"
    print(msg_start, flush=True)
    emit(msg_start, "step")
    
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("OPENAI_API_KEY")
    orig_key = os.environ.get("OPENAI_API_KEY")
    try:
        # Set key temporarily only during check
        if key:
            os.environ["OPENAI_API_KEY"] = key
        
        rails = get_rails_instance()
        logger.info(f"[Guardrails] Invoking NeMo Guardrails check for: '{query}'")
        
        # Invoke NeMo Guardrails synchronously
        response = rails.generate(messages=[{"role": "user", "content": query}])
        
        # Parse the response based on format
        content = ""
        if isinstance(response, list) and len(response) > 0:
            content = response[0].get("content", "")
        elif isinstance(response, dict):
            content = response.get("content", "")
        else:
            content = str(response)
            
        content = content.strip()
        
        # Check if the returned content is a refusal
        # NeMo returns refusal text if a flow matched "bot refuse request"
        refusal_markers = [
            "I cannot help you with that request",
            "not for cooking, coding, developer simulation",
            "I am designed to assist only with technical documentation",
        ]
        
        is_blocked = any(marker in content for marker in refusal_markers)
        
        if is_blocked:
            msg_block = f"❌ [Guardrails] NeMo Guardrails BLOCKED query '{query}' | Refusal response: '{content}'"
            print(msg_block, flush=True)
            emit(msg_block, "warning")
            return True, content
            
        msg_allow = f"✅ [Guardrails] NeMo Guardrails ALLOWED query '{query}'"
        print(msg_allow, flush=True)
        emit(msg_allow, "success")
        return False, None
        
    except Exception as e:
        msg_error = f"⚠️ [Guardrails] Error during guardrails check: {e}. Defaulting to allowing query."
        print(msg_error, flush=True)
        emit(msg_error, "warning")
        logger.error(f"[Guardrails] Error during guardrails check: {e}")
        return False, None
    finally:
        # Restore original key
        if orig_key is not None:
            os.environ["OPENAI_API_KEY"] = orig_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)
