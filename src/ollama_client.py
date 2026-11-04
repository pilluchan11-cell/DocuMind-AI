"""
Ollama Client Module for DocuMind AI.
Connects directly to the local Ollama REST API using http://localhost:11434/api/chat.
Verifies service availability, model existence, and provides both streaming and
synchronous response generation with robust error handling.
"""

from typing import List, Dict, Any, Generator, Tuple, Optional
import json
import requests


class OllamaClientError(Exception):
    """Custom exception raised when Ollama connectivity or generation fails."""
    pass


class OllamaClient:
    """
    Manages local Ollama LLM inference using the /api/chat endpoint.
    """

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3.2:3b"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.tags_endpoint = f"{self.base_url}/api/tags"

    def check_connection(self) -> Tuple[bool, str, List[str]]:
        """
        Tests whether the local Ollama daemon is reachable and lists installed models.

        Returns:
            Tuple of:
                - is_connected (bool): True if Ollama is running and responding.
                - message (str): Status description.
                - installed_models (List[str]): List of locally available model tags.
        """
        try:
            resp = requests.get(self.tags_endpoint, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return True, "Ollama service is online and operational.", models
            else:
                return False, f"Ollama returned HTTP {resp.status_code}: {resp.text}", []
        except requests.exceptions.ConnectionError:
            return (
                False,
                "Cannot connect to local Ollama. The Ollama service is not running on port 11434.\n"
                "Please start Ollama by opening a terminal and running:\n"
                "    ollama serve\n"
                "(or launch Ollama from your desktop/start menu).",
                []
            )
        except requests.exceptions.Timeout:
            return False, "Connection to Ollama timed out (5s limit exceeded).", []
        except Exception as e:
            return False, f"Error communicating with Ollama: {str(e)}", []

    def is_model_installed(self, model_name: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Checks if the requested model is downloaded and ready in Ollama.

        Returns:
            Tuple of (is_installed: bool, installed_models: List[str])
        """
        target = (model_name or self.default_model).lower()
        is_conn, _, installed = self.check_connection()
        if not is_conn:
            return False, []

        installed_lower = [m.lower() for m in installed]
        # Match exact tag ("llama3.2:3b") or base prefix ("llama3.2")
        found = any(
            target == m or target in m or m.startswith(target.split(":")[0])
            for m in installed_lower
        )
        return found, installed

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: int = 120,
        max_tokens: int = 500
    ) -> str:
        """
        Generates a non-streaming response via http://localhost:11434/api/chat.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Ollama model name (defaults to self.default_model).
            temperature: Sampling temperature (0.1 for high factual grounding).
            timeout: Maximum seconds to wait for response.
            max_tokens: Maximum tokens to generate (num_predict).

        Returns:
            Generated response string.

        Raises:
            OllamaClientError: If generation fails or service is unreachable.
        """
        active_model = model or self.default_model
        payload = {
            "model": active_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=timeout
            )
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "").strip()
                if not content:
                    raise OllamaClientError("Ollama returned an empty response.")
                return content
            elif response.status_code == 404:
                raise OllamaClientError(
                    f"Model '{active_model}' not found in local Ollama library.\n"
                    f"To install this model, open a terminal and run:\n"
                    f"    ollama pull {active_model}"
                )
            else:
                raise OllamaClientError(
                    f"Ollama chat error (HTTP {response.status_code}): {response.text}"
                )
        except requests.exceptions.ConnectionError:
            raise OllamaClientError(
                "Failed to connect to Ollama. Ensure Ollama is running locally via 'ollama serve'."
            )
        except requests.exceptions.Timeout:
            raise OllamaClientError(
                f"Ollama request timed out after {timeout} seconds. "
                "The local model took too long to generate a response."
            )
        except Exception as e:
            if isinstance(e, OllamaClientError):
                raise e
            raise OllamaClientError(f"Unexpected error during Ollama chat: {str(e)}")

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: int = 120,
        max_tokens: int = 500
    ) -> Generator[str, None, None]:
        """
        Streams token chunks iteratively from http://localhost:11434/api/chat.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Ollama model name.
            temperature: Sampling temperature.
            timeout: Request timeout.
            max_tokens: Maximum tokens to generate.

        Yields:
            Token strings as they are generated.
        """
        active_model = model or self.default_model
        payload = {
            "model": active_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": max_tokens
            }
        }

        try:
            with requests.post(
                self.chat_endpoint,
                json=payload,
                stream=True,
                timeout=timeout
            ) as response:
                if response.status_code == 404:
                    raise OllamaClientError(
                        f"Model '{active_model}' not found. Run: ollama pull {active_model}"
                    )
                elif response.status_code != 200:
                    raise OllamaClientError(
                        f"Ollama stream failed with status {response.status_code}: {response.text}"
                    )

                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        msg = chunk.get("message", {})
                        token = msg.get("content", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
        except requests.exceptions.ConnectionError:
            raise OllamaClientError("Connection to Ollama was lost during streaming generation.")
        except Exception as e:
            if isinstance(e, OllamaClientError):
                raise e
            raise OllamaClientError(f"Streaming error: {str(e)}")
