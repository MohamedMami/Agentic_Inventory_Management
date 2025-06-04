import logging
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Literal, Union, Annotated
from enum import Enum
from functools import lru_cache
from sqlalchemy.orm import Session
from datetime import datetime
import json
import os
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field  # Updated import
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

# Configuration du logging
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query types supported by the system."""
    INVENTORY = "inventory"
    VISUALIZATION = "visualization"
    FORECAST = "forecast"
    UNKNOWN = "unknown"
    COMPOSITE = "composite"  # New type for multi-part queries

# Define state types for the ReAct agent
class Thought(TypedDict):
    reasoning: str

class Action(TypedDict, total=False):
    action_type: Literal["classify_query", "process_inventory", "process_visualization", 
                        "process_forecast", "composite_planning", "final_response"]
    query_type: Optional[str]
    explanation: Optional[str]
    sub_queries: Optional[List[Dict[str, Any]]]

class ActionResult(TypedDict, total=False):
    status: Literal["success", "error"]
    data: Optional[Any]
    error: Optional[str]
    agent_response: Optional[Dict[str, Any]]

class MemoryEntry(TypedDict):
    timestamp: str
    query: str
    query_type: str
    response: Optional[Dict[str, Any]]

class AgentState(TypedDict):
    query: str
    session: Session
    thoughts: List[Thought]
    actions: List[Action]
    action_results: List[ActionResult]
    query_type: Optional[str]
    sub_queries: Optional[List[Dict[str, Any]]]
    final_response: Optional[Dict[str, Any]]
    memory: List[MemoryEntry]
    conversation_id: str
    chat_history: Optional[List[Dict[str, Any]]]

# Output models for each step
class QueryClassificationOutput(BaseModel):
    thought: str = Field(description="Your reasoning about this query type")
    query_type: str = Field(description="The classified query type")
    is_composite: bool = Field(description="Whether this query requires multiple specialized agents")
    sub_queries: Optional[List[Dict[str, str]]] = Field(None, 
        description="If composite, break down into sub-queries with their types")
    explanation: str = Field(description="Explanation for the classification")

class FinalResponseOutput(BaseModel):
    response: Dict[str, Any] = Field(description="Final structured response to return")

class SupervisorReActAgent:
    """
    Enhanced SupervisorAgent with ReAct pattern, memory and handoffs implemented with LangGraph.
    """
    
    # Add QueryType as a class attribute
    QueryType = QueryType
    
    def __init__(self, model_name="llama-3.3-70b-versatile", temperature=0, redis_url="redis://localhost:6379"):
        """Initialize the ReAct Supervisor agent with LangGraph components"""
        # Check for GROQ API key
        groq_api_key = os.getenv("groq_api_key")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        # Validate Redis URL
        if not redis_url:
            redis_url = "redis://localhost:6379"  # Default fallback
        
        # Initialize the LLM
        try:
            self.llm = ChatGroq(
                temperature=temperature,
                groq_api_key=groq_api_key,
                model_name=model_name
            )
            
            # Store validated Redis URL
            self.redis_url = redis_url
            
            # Initialize rest of components
            # Query patterns for classification
            self.query_patterns = {
                QueryType.INVENTORY: [
                    "stock", "inventory", "quantity", "available", "warehouse",
                    "expiry", "batch", "supply", "units", "products"
                ],
                QueryType.VISUALIZATION: [
                    "show", "display", "graph", "chart", "plot", "visualize",
                    "trend", "compare", "distribution", "breakdown"
                ],
                QueryType.FORECAST: [
                    "predict", "forecast", "future", "projection", "estimate",
                    "demand", "next", "upcoming", "expected", "trend"
                ]
            }
            
            # Create output parsers
            self.classification_parser = JsonOutputParser(pydantic_object=QueryClassificationOutput)
            self.response_parser = JsonOutputParser(pydantic_object=FinalResponseOutput)
            
            # Build the ReAct graph
            self.workflow = self._build_graph()
            
            # Agent registry for handoffs - would be instantiated by actual code
            # This is just a placeholder structure
            self.agent_registry = {
                "inventory": None,  # Will be set from outside
                "visualization": None,  # Will be set from outside
                "forecast": None  # Will be set from outside
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize SupervisorReActAgent: {str(e)}")
            raise
    
    def register_agent(self, agent_type: str, agent_instance: Any) -> None:
        """Register a specialized agent for handoffs"""
        self.agent_registry[agent_type.lower()] = agent_instance
        logger.info(f"Registered {agent_type} agent")
    
    def _get_memory(self, conversation_id: str) -> BaseChatMessageHistory:
        """Get or create a memory store for this conversation"""
        return RedisChatMessageHistory(
            url=self.redis_url,
            ttl=60 * 60 * 24,  # 24 hour expiration
            session_id=f"supervisor:{conversation_id}"
        )
    
    def _build_graph(self) -> Pregel:
        """Build the LangGraph workflow for the ReAct Supervisor agent"""
        
        # 1. Create the graph
        graph = StateGraph(AgentState)
        
        # 2. Add nodes with async support
        graph.add_node("classify_query", self._classify_query)
        graph.add_node("process_inventory", self._process_inventory_query)
        graph.add_node("process_visualization", self._process_visualization_query)
        graph.add_node("process_forecast", self._process_forecast_query)
        graph.add_node("process_composite", self._process_composite_query)
        graph.add_node("generate_final_response", self._generate_final_response)
        graph.add_node("update_memory", self._update_memory)
        
        # 3. Add edges
        # From classification to appropriate processing
        graph.add_conditional_edges(
            "classify_query",
            self._route_after_classification,
            {
                "inventory": "process_inventory",
                "visualization": "process_visualization",
                "forecast": "process_forecast",
                "composite": "process_composite",
                "unknown": "generate_final_response"
            }
        )
        
        # From processing to final response
        graph.add_edge("process_inventory", "generate_final_response")
        graph.add_edge("process_visualization", "generate_final_response")
        graph.add_edge("process_forecast", "generate_final_response")
        graph.add_edge("process_composite", "generate_final_response")
        
        # From final response to memory update
        graph.add_edge("generate_final_response", "update_memory")
        
        # From memory update to end
        graph.add_edge("update_memory", END)
        
        # 4. Set the entry point
        graph.set_entry_point("classify_query")
        
        return graph.compile()
    
    def _route_after_classification(self, state: AgentState) -> str:
        """Decide routing based on query classification"""
        query_type = state.get("query_type", "unknown").lower()
        
        if query_type == "composite":
            return "composite"
        elif query_type in ["inventory", "visualization", "forecast"]:
            return query_type
        else:
            return "unknown"
    
    def _classify_query(self, state: AgentState) -> AgentState:
        """Classify the query type"""
        query = state["query"]
        
        # Define the classification prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI agent that classifies queries into specific types. 
            Output MUST be in JSON format with the following fields:
            {
                "thought": "your reasoning",
                "query_type": "one of: INVENTORY, VISUALIZATION, FORECAST, COMPOSITE, UNKNOWN",
                "is_composite": true/false,
                "sub_queries": [] (optional, only if composite),
                "explanation": "brief explanation"
            }"""),
            HumanMessage(content=f"Classify this query: {query}")
        ])
        
        # Create the classification chain
        chain = prompt | self.llm | self.classification_parser
        
        try:
            # Get the classification result as a dictionary
            result = chain.invoke({"query": query})
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
            
            return {
                **state,
                "query_type": result_dict.get("query_type", "UNKNOWN"),
                "sub_queries": result_dict.get("sub_queries") if result_dict.get("is_composite") else None,
                "thoughts": state["thoughts"] + [{"reasoning": result_dict.get("thought", "")}]
            }
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {
                **state,
                "query_type": "UNKNOWN",
                "error": f"Failed to classify query: {str(e)}"
            }
    
    def _process_inventory_query(self, state: AgentState) -> AgentState:
        """Process query using the inventory agent"""
        query = state["query"]
        session = state["session"]
        
        new_state = state.copy()
        
        # Get the appropriate agent
        inventory_agent = self.agent_registry.get("inventory")
        
        if not inventory_agent:
            # No agent registered, return error
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "error",
                "error": "Inventory agent not registered"
            }]
            return new_state
        
        try:
            # Hand off to the inventory agent
            agent_response = inventory_agent.process_query(query, session)
            
            # Update state with result
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "success",
                "agent_response": agent_response
            }]
        
        except Exception as e:
            logger.error(f"Error in inventory agent: {e}")
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "error",
                "error": f"Error in inventory agent: {str(e)}"
            }]
        
        return new_state
    
    async def _process_visualization_query(self, state: AgentState) -> AgentState:
        """Process query using the visualization agent"""
        query = state["query"]
        session = state["session"]
        
        new_state = state.copy()
        
        # Get the appropriate agent
        visualization_agent = self.agent_registry.get("visualization")
        
        if not visualization_agent:
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "error",
                "error": "Visualization agent not registered"
            }]
            return new_state
        
        try:
            # Hand off to the visualization agent and properly await the response
            if asyncio.iscoroutinefunction(visualization_agent.process_query):
                agent_response = await visualization_agent.process_query(query, session)
            else:
                agent_response = visualization_agent.process_query(query, session)
            
            # Update state with result
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "success",
                "agent_response": agent_response
            }]
        
        except Exception as e:
            logger.error(f"Error in visualization agent: {e}")
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "error",
                "error": f"Error in visualization agent: {str(e)}"
            }]
        
        return new_state
    
    def _process_forecast_query(self, state: AgentState) -> AgentState:
        """Process query using the forecast agent"""
        query = state["query"]
        session = state["session"]
        
        new_state = state.copy()
        
        # Get the appropriate agent
        forecast_agent = self.agent_registry.get("forecast")
        
        if not forecast_agent:
            # No agent registered, return error
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "error",
                "error": "Forecast agent not registered"
            }]
            return new_state
        
        try:
            # Hand off to the forecast agent
            agent_response = forecast_agent.process_query(query, session)
            
            # Update state with result
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "success",
                "agent_response": agent_response
            }]
        
        except Exception as e:
            logger.error(f"Error in forecast agent: {e}")
            new_state["action_results"] = state.get("action_results", []) + [{
                "status": "error",
                "error": f"Error in forecast agent: {str(e)}"
            }]
        
        return new_state
    
    async def _process_composite_query(self, state: AgentState) -> AgentState:
        """Process a composite query by breaking it down and handling sub-queries"""
        try:
            sub_responses = []
            for sub_query in state["sub_queries"]:
                # Process each sub-query
                sub_result = await self._process_sub_query(sub_query, state["session"])
                sub_responses.append({
                    "query_type": sub_query["type"],
                    "query": sub_query["query"],
                    "result": sub_result
                })

            # Update state with composite results
            state["final_response"] = {
                "integrated_response": "Composite query processed successfully",
                "sub_responses": sub_responses,
                "data": None  # Add any aggregated data if needed
            }
            
            return state

        except Exception as e:
            logger.error(f"Error in composite query processing: {str(e)}")
            state["final_response"] = {
                "response": f"Error processing composite query: {str(e)}",
                "error": str(e)
            }
            return state
    async def _process_sub_query(self, sub_query: Dict[str, str], session: Session) -> Dict[str, Any]:
        """Process an individual sub-query by routing to appropriate agent"""
        query_type = sub_query.get("type", "").lower()
        query_text = sub_query.get("query", "")
        
        try:
            # Get the appropriate agent
            agent = self.agent_registry.get(query_type)
            
            if not agent:
                return {
                    "status": "error",
                    "error": f"No agent registered for query type: {query_type}",
                    "query": query_text
                }
            
            # Process with appropriate agent
            if hasattr(agent, 'process_query'):
                if asyncio.iscoroutinefunction(agent.process_query):
                    result = await agent.process_query(query_text, session)
                else:
                    result = agent.process_query(query_text, session)
                    
                return {
                    "status": "success",
                    "result": result,
                    "query": query_text,
                    "query_type": query_type
                }
            else:
                return {
                    "status": "error",
                    "error": f"Agent for {query_type} does not implement process_query",
                    "query": query_text
                }
                
        except Exception as e:
            logger.error(f"Error processing sub-query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "query": query_text,
                "query_type": query_type
            }
        
    def _generate_final_response(self, state: AgentState) -> AgentState:
        """Generate the final response by integrating all results"""
        query = state["query"]
        query_type = state["query_type"]
        action_results = state["action_results"]
        
        new_state = state.copy()
        
        # If there are no results, return an error
        if not action_results:
            response = {
                "query": query,
                "query_type": query_type,
                "timestamp": datetime.now().isoformat(),
                "response": "No response could be generated.",
                "error": "No action results available"
            }
            new_state["final_response"] = response
            return new_state
        
        # For composite queries, generate an integrated response
        if query_type == "composite":
            # Get the last action result
            last_result = action_results[-1]
            sub_query_results = last_result.get("sub_query_results", [])
            
            # Format integrated response
            success_results = [r for r in sub_query_results if r.get("status") == "success"]
            error_results = [r for r in sub_query_results if r.get("status") != "success"]
            
            # Build the final response
            response = {
                "query": query,
                "query_type": query_type,
                "timestamp": datetime.now().isoformat(),
                "sub_responses": success_results,
                "errors": [r.get("error") for r in error_results] if error_results else None,
                "integrated_response": self._integrate_composite_results(query, sub_query_results)
            }
        
        # For single agent queries
        else:
            # Get the last action result
            last_result = action_results[-1]
            
            # Check if there was an error
            if last_result.get("status") == "error":
                response = {
                    "query": query,
                    "query_type": query_type,
                    "timestamp": datetime.now().isoformat(),
                    "error": last_result.get("error"),
                    "response": f"An error occurred: {last_result.get('error')}"
                }
            else:
                # Get the agent response
                agent_response = last_result.get("agent_response", {})
                
                # Build the final response
                response = {
                    "query": query,
                    "query_type": query_type,
                    "timestamp": datetime.now().isoformat(),
                    "response": agent_response.get("response"),
                    "data": agent_response.get("data"),
                    "visualization": agent_response.get("visualization"),
                    "visualization_path": agent_response.get("visualization_path")  if agent_response.get("visualization_path") else None,
                    "visualization_base64": agent_response.get("visualization_base64")  if agent_response.get("visualization_base64") else None,
                    "error": agent_response.get("error")
                }
        
        # Update state
        new_state["final_response"] = response
        
        return new_state
    
    def _integrate_composite_results(self, query: str, sub_query_results: List[Dict[str, Any]]) -> str:
        """Integrate results from multiple agents into a coherent response"""
        # Convert results to a format for LLM processing
        results_json = json.dumps(sub_query_results, default=str)
        
        # Build prompt for integration
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in pharmaceutical inventory management.
            Your task is to integrate results from multiple specialized agents into a coherent response.
            Be clear, concise, and make sure all relevant information is included in a well-structured way.
            """),
            ("human", """
            Original user query: {query}
            
            Results from various agents: {results}
            
            Please provide an integrated response that addresses the original query
            using information from all the specialized agents.
            """)
        ])
        
        # Execute chain
        chain = prompt | self.llm | StrOutputParser()
        integrated_response = chain.invoke({
            "query": query,
            "results": results_json
        })
        
        return integrated_response
    
    def _update_memory(self, state: AgentState) -> AgentState:
        """Update memory with the current interaction"""
        query = state["query"]
        query_type = state["query_type"]
        conversation_id = state["conversation_id"]
        final_response = state["final_response"]
        
        # Get the memory for this conversation
        memory = self._get_memory(conversation_id)
        
        # Create a memory entry
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_type": query_type,
            "response": final_response
        }
        
        # Add to memory history in state
        new_state = state.copy()
        new_state["memory"] = state.get("memory", []) + [memory_entry]
        
        # Add messages to Redis memory for long-term storage
        memory.add_user_message(query)
        memory.add_ai_message(json.dumps(final_response, default=str))
        
        return new_state
    
    def _load_chat_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Load chat history from memory"""
        memory = self._get_memory(conversation_id)
        messages = memory.messages
        
        # Convert to a list of dictionaries
        history = []
        for message in messages:
            history.append({
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content
            })
        
        return history
    
    async def process_query(
        self,
        query: str,
        session: Session,
        conversation_id: str = None
    ) -> Dict[str, Any]:
        """Process a query and return structured response"""
        try:
            # Initialize state
            state = AgentState(
                query=query,
                session=session,
                thoughts=[],
                actions=[],
                action_results=[],
                query_type=None,
                sub_queries=None,
                final_response=None,
                memory=[],
                conversation_id=conversation_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                chat_history=[]
            )

            # Run the workflow using ainvoke instead of arun
            final_state = await self.workflow.ainvoke(state)

            # Structure the response
            response = {
                "query": query,
                "query_type": final_state["query_type"],
                "response": final_state.get("final_response", {}).get("response", "No response generated"),
                "data": final_state.get("final_response", {}).get("data"),
                "visualization_path": final_state.get("final_response", {}).get("visualization_path"),
                "error": None
            }

            # Handle composite queries
            if final_state["query_type"] == "composite":
                response.update({
                    "integrated_response": final_state.get("final_response", {}).get("integrated_response"),
                    "sub_responses": final_state.get("final_response", {}).get("sub_responses", [])
                })

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "query_type": "unknown",
                "response": "An error occurred while processing the query.",
                "error": str(e),
                "data": None
            }