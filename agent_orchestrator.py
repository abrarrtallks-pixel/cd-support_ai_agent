from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import json

class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    ESCALATING = "escalating"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ConversationContext:
    conversation_id: str
    user_id: str
    session_id: str
    messages: List[Dict]
    user_profile: Dict
    sentiment_score: float
    escalation_score: float
    metadata: Dict
    created_at: datetime
    updated_at: datetime

class AgentOrchestrator:
    """
    Central orchestration engine for the AI support agent.
    Coordinates all components and manages conversation flow.
    """
    
    def __init__(
        self,
        llm_service,
        memory_service,
        rag_service,
        tool_registry,
        sentiment_analyzer,
        guardrail_service,
        escalation_service,
        analytics_service
    ):
        self.llm = llm_service
        self.memory = memory_service
        self.rag = rag_service
        self.tools = tool_registry
        self.sentiment = sentiment_analyzer
        self.guardrails = guardrail_service
        self.escalation = escalation_service
        self.analytics = analytics_service
        self.state = AgentState.IDLE
        
    async def process_message(
        self, 
        user_message: str, 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Main processing pipeline for incoming user messages.
        """
        try:
            self.state = AgentState.PROCESSING
            
            # Step 1: Update conversation context
            await self.memory.add_message(
                context.conversation_id,
                role="user",
                content=user_message
            )
            
            # Step 2: Sentiment Analysis (parallel)
            sentiment_task = asyncio.create_task(
                self.sentiment.analyze(user_message, context)
            )
            
            # Step 3: Retrieve relevant knowledge (RAG)
            rag_task = asyncio.create_task(
                self.rag.retrieve_relevant_context(
                    query=user_message,
                    conversation_history=context.messages[-5:],  # Last 5 messages
                    user_profile=context.user_profile
                )
            )
            
            # Wait for parallel tasks
            sentiment_result, rag_context = await asyncio.gather(
                sentiment_task, rag_task
            )
            
            # Update context with sentiment
            context.sentiment_score = sentiment_result['score']
            context.escalation_score = sentiment_result['escalation_probability']
            
            # Step 4: Check for immediate escalation
            if await self._should_escalate(context, sentiment_result):
                return await self._handle_escalation(context, sentiment_result)
            
            # Step 5: Build enriched prompt
            enriched_prompt = await self._build_prompt(
                user_message=user_message,
                context=context,
                rag_context=rag_context,
                sentiment=sentiment_result
            )
            
            # Step 6: Get LLM response with tool calling
            llm_response = await self._get_llm_response_with_tools(
                enriched_prompt, 
                context
            )
            
            # Step 7: Apply guardrails
            filtered_response = await self.guardrails.validate_response(
                llm_response, 
                context
            )
            
            # Step 8: Check if hallucination detected
            if filtered_response.get('hallucination_detected'):
                # Retry with stronger constraints
                llm_response = await self._retry_with_constraints(
                    enriched_prompt, 
                    context,
                    previous_response=llm_response
                )
                filtered_response = await self.guardrails.validate_response(
                    llm_response, 
                    context
                )
            
            # Step 9: Store assistant message
            await self.memory.add_message(
                context.conversation_id,
                role="assistant",
                content=filtered_response['content'],
                metadata={
                    'sentiment': sentiment_result,
                    'tools_used': filtered_response.get('tools_used', []),
                    'rag_sources': rag_context.get('sources', []),
                    'confidence': filtered_response.get('confidence', 1.0)
                }
            )
            
            # Step 10: Log analytics
            await self.analytics.track_interaction(
                context=context,
                user_message=user_message,
                response=filtered_response,
                sentiment=sentiment_result,
                duration=datetime.now() - context.updated_at
            )
            
            self.state = AgentState.IDLE
            
            return {
                'success': True,
                'response': filtered_response['content'],
                'sentiment': sentiment_result,
                'requires_followup': filtered_response.get('requires_followup', False),
                'metadata': {
                    'tools_used': filtered_response.get('tools_used', []),
                    'sources': rag_context.get('sources', []),
                    'confidence': filtered_response.get('confidence', 1.0)
                }
            }
            
        except Exception as e:
            self.state = AgentState.ERROR
            await self._handle_error(e, context)
            return {
                'success': False,
                'error': str(e),
                'fallback_response': await self._get_fallback_response(context)
            }
    
    async def _build_prompt(
        self, 
        user_message: str,
        context: ConversationContext,
        rag_context: Dict,
        sentiment: Dict
    ) -> str:
        """
        Constructs comprehensive prompt with all context.
        """
        # Get conversation history
        history = await self.memory.get_conversation_history(
            context.conversation_id,
            limit=10
        )
        
        # Format conversation history
        formatted_history = self._format_conversation_history(history)
        
        # Extract user profile information
        user_info = self._format_user_profile(context.user_profile)
        
        # Format RAG context
        knowledge_context = self._format_rag_context(rag_context)
        
        # Build comprehensive prompt
        prompt = f"""You are an expert customer support agent for our company. Your goal is to provide helpful, accurate, and empathetic support.

## CONVERSATION CONTEXT
{formatted_history}

## USER PROFILE
{user_info}

## RELEVANT KNOWLEDGE BASE
{knowledge_context}

## CURRENT SENTIMENT ANALYSIS
User Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})
Emotion Detected: {sentiment.get('emotion', 'neutral')}
Escalation Risk: {sentiment['escalation_probability']:.2%}

## AVAILABLE TOOLS
You have access to the following tools to help the user:
{self._format_available_tools()}

## GUIDELINES
1. Be empathetic and acknowledge the user's feelings, especially if sentiment is negative
2. Provide accurate information from the knowledge base
3. Use tools when necessary to fetch real-time data or perform actions
4. If you're unsure, admit it and offer to escalate to a human agent
5. Keep responses concise but complete
6. Always verify information before stating it as fact
7. Protect user privacy - never expose sensitive data
8. If the user is frustrated (negative sentiment), prioritize de-escalation

## CURRENT USER MESSAGE
{user_message}

Please provide a helpful response. If you need to use any tools, indicate which ones and why."""

        return prompt
    
    async def _get_llm_response_with_tools(
        self, 
        prompt: str, 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Gets LLM response and handles tool calling iteratively.
        """
        max_iterations = 5
        iteration = 0
        tools_used = []
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.llm.generate(
                prompt=prompt,
                tools=self.tools.get_tool_definitions(),
                temperature=0.3,  # Lower temp for consistency
                max_tokens=1000
            )
            
            # Check if tool calls are requested
            if not response.get('tool_calls'):
                # No more tool calls, return final response
                return {
                    'content': response['content'],
                    'tools_used': tools_used,
                    'confidence': response.get('confidence', 1.0)
                }
            
            # Execute tool calls
            tool_results = []
            for tool_call in response['tool_calls']:
                try:
                    self.state = AgentState.WAITING_FOR_TOOL
                    
                    result = await self.tools.execute(
                        tool_name=tool_call['name'],
                        parameters=tool_call['parameters'],
                        context=context
                    )
                    
                    tool_results.append({
                        'tool': tool_call['name'],
                        'result': result,
                        'success': True
                    })
                    tools_used.append(tool_call['name'])
                    
                except Exception as e:
                    tool_results.append({
                        'tool': tool_call['name'],
                        'error': str(e),
                        'success': False
                    })
            
            # Update prompt with tool results for next iteration
            prompt = self._append_tool_results_to_prompt(
                prompt, 
                response['content'], 
                tool_results
            )
            
            self.state = AgentState.PROCESSING
        
        # Max iterations reached
        return {
            'content': response['content'],
            'tools_used': tools_used,
            'warning': 'Max tool iterations reached'
        }
    
    async def _should_escalate(
        self, 
        context: ConversationContext, 
        sentiment: Dict
    ) -> bool:
        """
        Determines if conversation should be escalated to human.
        """
        escalation_triggers = [
            # High negative sentiment
            sentiment['score'] < -0.7,
            # Explicit escalation request
            any(keyword in context.messages[-1]['content'].lower() 
                for keyword in ['speak to human', 'manager', 'supervisor', 'escalate']),
            # High escalation probability from sentiment analysis
            sentiment.get('escalation_probability', 0) > 0.8,
            # Too many back-and-forth messages without resolution
            len(context.messages) > 15 and context.escalation_score > 0.5,
            # VIP customer with issue
            context.user_profile.get('tier') == 'VIP' and sentiment['score'] < 0,
        ]
        
        return any(escalation_triggers)
    
    async def _handle_escalation(
        self, 
        context: ConversationContext,
        sentiment: Dict
    ) -> Dict[str, Any]:
        """
        Handles escalation to human agent.
        """
        self.state = AgentState.ESCALATING
        
        escalation_result = await self.escalation.create_ticket(
            context=context,
            priority=self._calculate_priority(context, sentiment),
            reason=self._generate_escalation_reason(context, sentiment),
            sentiment=sentiment
        )
        
        return {
            'success': True,
            'escalated': True,
            'response': escalation_result['message_to_user'],
            'ticket_id': escalation_result['ticket_id'],
            'estimated_wait_time': escalation_result['estimated_wait_time']
        }
    
    def _calculate_priority(
        self, 
        context: ConversationContext, 
        sentiment: Dict
    ) -> str:
        """
        Calculates escalation priority (URGENT, HIGH, MEDIUM, LOW).
        """
        score = 0
        
        # Sentiment weight
        if sentiment['score'] < -0.7:
            score += 3
        elif sentiment['score'] < -0.3:
            score += 2
        
        # User tier weight
        tier_weights = {'VIP': 3, 'Premium': 2, 'Standard': 1}
        score += tier_weights.get(context.user_profile.get('tier', 'Standard'), 1)
        
        # Conversation length weight
        if len(context.messages) > 20:
            score += 2
        elif len(context.messages) > 10:
            score += 1
        
        # Return priority
        if score >= 7:
            return 'URGENT'
        elif score >= 5:
            return 'HIGH'
        elif score >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Formats conversation history for prompt."""
        formatted = []
        for msg in history:
            role = msg['role'].upper()
            content = msg['content']
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    def _format_user_profile(self, profile: Dict) -> str:
        """Formats user profile for prompt."""
        return f"""
User ID: {profile.get('user_id', 'N/A')}
Name: {profile.get('name', 'N/A')}
Tier: {profile.get('tier', 'Standard')}
Account Created: {profile.get('created_at', 'N/A')}
Total Orders: {profile.get('total_orders', 0)}
Lifetime Value: ${profile.get('lifetime_value', 0)}
Language Preference: {profile.get('language', 'en')}
"""
    
    def _format_rag_context(self, rag_context: Dict) -> str:
        """Formats RAG context for prompt."""
        if not rag_context.get('documents'):
            return "No relevant knowledge base articles found."
        
        formatted = []
        for idx, doc in enumerate(rag_context['documents'][:3], 1):  # Top 3
            formatted.append(f"""
Document {idx} (Relevance: {doc['score']:.2f}):
Title: {doc['title']}
Content: {doc['content']}
Source: {doc['source']}
""")
        return "\n".join(formatted)
    
    def _format_available_tools(self) -> str:
        """Formats available tools for prompt."""
        tools = self.tools.get_tool_definitions()
        formatted = []
        for tool in tools:
            formatted.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(formatted)
    
    async def _handle_error(self, error: Exception, context: ConversationContext):
        """Handles errors gracefully."""
        # Log error
        await self.analytics.track_error(
            error=error,
            context=context,
            component='AgentOrchestrator'
        )
        
        # Notify monitoring
        # Send to Sentry/DataDog
        pass
    
    async def _get_fallback_response(self, context: ConversationContext) -> str:
        """Provides fallback response when system fails."""
        return """I apologize, but I'm experiencing technical difficulties right now. 
Let me connect you with a human agent who can help you immediately. 
Your conversation has been saved and they'll have full context."""