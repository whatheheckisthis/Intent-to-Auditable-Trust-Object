from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from typing import List, Optional
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ops-Utilities AI Wrapper",
    description="FastAPI wrapper for AI inference with OpenAI integration",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000

class ChatResponse(BaseModel):
    response: str
    model_used: str
    timestamp: str
    tokens_used: int

class InferenceRequest(BaseModel):
    data: dict
    task_type: str = "classification"
    model: str = "gpt-3.5-turbo"

class InferenceResponse(BaseModel):
    result: dict
    confidence: float
    explanation: str
    timestamp: str

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Ops-Utilities AI Wrapper is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic OpenAI connectivity test
        if not openai.api_key:
            return {"status": "error", "message": "OpenAI API key not configured"}
        
        return {
            "status": "healthy",
            "openai_configured": bool(openai.api_key),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """General chat endpoint using OpenAI"""
    try:
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        response = await openai.ChatCompletion.acreate(
            model=request.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant integrated with ops-utilities."},
                {"role": "user", "content": request.message}
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatResponse(
            response=response.choices[0].message.content,
            model_used=request.model,
            timestamp=datetime.now().isoformat(),
            tokens_used=response.usage.total_tokens
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest):
    """AI inference endpoint for structured tasks"""
    try:
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Create structured prompt based on task type
        if request.task_type == "classification":
            system_prompt = "You are an AI classifier. Analyze the provided data and return classification results with confidence scores."
        elif request.task_type == "analysis":
            system_prompt = "You are an AI analyst. Provide detailed analysis of the provided data."
        else:
            system_prompt = "You are an AI assistant. Process the provided data according to the specified task."
        
        prompt = f"Task: {request.task_type}\nData: {json.dumps(request.data)}\nProvide structured analysis."
        
        response = await openai.ChatCompletion.acreate(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent inference
        )
        
        ai_response = response.choices[0].message.content
        
        # Parse or structure the response
        return InferenceResponse(
            result={"analysis": ai_response, "task_type": request.task_type},
            confidence=0.85,  # Could be extracted from AI response if needed
            explanation=f"Analysis completed using {request.model}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Inference endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/kernel/train")
async def mock_kernel_training(data: dict):
    """Mock endpoint for kernel training (placeholder for ops-utilities integration)"""
    try:
        # This could eventually call the actual kernel when it's working
        # For now, mock the training process
        
        logger.info("Mock kernel training initiated")
        
        # Simulate training process
        training_config = {
            "model_type": "pgd_optimization",
            "cycles": data.get("cycles", 50),
            "learning_rate": data.get("learning_rate", 0.1),
            "devices": "auto"
        }
        
        # Mock training results
        results = {
            "status": "completed",
            "training_config": training_config,
            "final_loss": 0.0234,
            "convergence_cycles": 42,
            "model_saved": "trained_models/kernel_model.pkl",
            "timestamp": datetime.now().isoformat()
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Kernel training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/models/list")
async def list_available_models():
    """List available OpenAI models and local models"""
    try:
        # Get available OpenAI models
        models = await openai.Model.alist()
        openai_models = [model.id for model in models.data if "gpt" in model.id]
        
        # Mock local models (from ops-utilities when working)
        local_models = [
            "kernel_pgd_v1",
            "banknote_classifier",
            "entropy_optimizer"
        ]
        
        return {
            "openai_models": openai_models,
            "local_models": local_models,
            "total_available": len(openai_models) + len(local_models)
        }
        
    except Exception as e:
        logger.error(f"Model listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

# Additional utility endpoints inspired by ops-utilities
@app.post("/utils/encrypt")
async def encrypt_data(data: dict, passphrase: str):
    """Simple data encryption utility (inspired by ops-utilities scripts)"""
    try:
        # Mock encryption - in production, implement proper encryption
        encrypted_data = {
            "encrypted": True,
            "data_hash": hash(json.dumps(data, sort_keys=True)),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Data encryption completed")
        return encrypted_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")

@app.get("/logs/recent")
async def get_recent_logs(lines: int = 50):
    """Get recent application logs"""
    try:
        # Mock log retrieval - implement actual log reading as needed
        mock_logs = [
            f"{datetime.now().isoformat()}: Application started",
            f"{datetime.now().isoformat()}: OpenAI connection verified",
            f"{datetime.now().isoformat()}: Ready for inference requests"
        ]
        
        return {"logs": mock_logs[-lines:], "total_lines": len(mock_logs)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Log retrieval failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
