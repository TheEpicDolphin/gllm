import asyncio
from dataclasses import dataclass

from fastapi import FastAPI

from server import GenerationRequest, GenerationResult, Server
    
    
class NetworkedServerWrapper:
    def __init__(self, server: Server):
        self.server = server
        self.app = FastAPI()
        self._setup_routes()


    def _setup_routes(self):
        @self.app.post("/generate")
        async def generate(request: Request):
            # Get raw JSON body.
            data = await request.json()

            # Basic validation / defaults
            prompt = data.get("prompt")
            max_new_tokens = data.get("max_new_tokens", 32)
            stop_tokens = data.get("stop_tokens", [])
            temperature = data.get("temperature", 1.0)
            top_p = data.get("top_p", 1.0)
            top_k = data.get("top_k", 0)
            
            # Build generation request.
            internal_req = GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
                temperature=req.temperature,
                top_k=top_k,
                top_p=top_p,
            )
            # Enqueue and wait for result
            result = await self.server.enqueue_request(internal_req)
            return {
                "text": result.text,
            }
