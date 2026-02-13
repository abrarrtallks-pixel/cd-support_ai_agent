import anthropic
import openai
from typing import Dict, List, Optional, Any
import asyncio
from functools import wraps
import time

class LLMService:
    """
    Handles all LLM interactions with failover and retry logic.
    """
    
    def __init__(self, config: Dict):
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=config['anthropic_api_key']
        )
        self.openai_client = openai.AsyncOpenAI(
            api_key=config['openai_api_key']
        )
        self.primary_model = config.get('primary_model', 'claude-sonnet-4-20250514')
        self.fallback_model = config.get('fallback_model', 'gpt-4-turbo-preview')
        self.max_retries = 3
        self.timeout = 30
        
    async def generate(
        self,
        prompt: str,
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response with automatic failover.
        """
        # Try primary (Claude)
        try:
            return await self._generate_claude(
                prompt=prompt,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
        except Exception as e:
            print(f"Claude failed: {e}. Failing over to GPT-4...")
            
            # Fallback to GPT-4
            try:
                return await self._generate_openai(
                    prompt=prompt,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e2:
                print(f"GPT-4 also failed: {e2}")
                raise Exception("All LLM providers failed")
    
    async def _generate_claude(
        self,
        prompt: str,
        tools: Optional[List[Dict]],
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate using Claude with tool support.
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Convert tools to Claude format
        claude_tools = self._convert_tools_to_claude_format(tools) if tools else None
        
        response = await self.anthropic_client.messages.create(
            model=self.primary_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "You are a helpful customer support agent.",
            messages=messages,
            tools=claude_tools
        )
        
        # Parse response
        content = ""
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    'name': block.name,
                    'parameters': block.input,
                    'id': block.id
                })
        
        return {
            'content': content,
            'tool_calls': tool_calls if tool_calls else None,
            'model': self.primary_model,
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }
        }
    
    async def _generate_openai(
        self,
        prompt: str,
        tools: Optional[List[Dict]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Generate using OpenAI with tool support.
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Convert tools to OpenAI format
        openai_tools = self._convert_tools_to_openai_format(tools) if tools else None
        
        kwargs = {
            'model': self.fallback_model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        if openai_tools:
            kwargs['tools'] = openai_tools
            kwargs['tool_choice'] = 'auto'
        
        response = await self.openai_client.chat.completions.create(**kwargs)
        
        message = response.choices[0].message
        
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append({
                    'name': tool_call.function.name,
                    'parameters': json.loads(tool_call.function.arguments),
                    'id': tool_call.id
                })
        
        return {
            'content': message.content or "",
            'tool_calls': tool_calls if tool_calls else None,
            'model': self.fallback_model,
            'usage': {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            }
        }
    
    def _convert_tools_to_claude_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert generic tool definitions to Claude format."""
        claude_tools = []
        for tool in tools:
            claude_tools.append({
                'name': tool['name'],
                'description': tool['description'],
                'input_schema': {
                    'type': 'object',
                    'properties': tool['parameters'],
                    'required': tool.get('required', [])
                }
            })
        return claude_tools
    
    def _convert_tools_to_openai_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert generic tool definitions to OpenAI format."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                'type': 'function',
                'function': {
                    'name': tool['name'],
                    'description': tool['description'],
                    'parameters': {
                        'type': 'object',
                        'properties': tool['parameters'],
                        'required': tool.get('required', [])
                    }
                }
            })
        return openai_tools