from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from app.graph.workflow import app_graph
import json
import asyncio

app = FastAPI(title="FinAgent API")

@app.get("/")
async def root():
    return {"status": "online", "engine": "LangGraph-Financial-Reasoning"}

@app.websocket("/api/v1/analyze")
async def analyze_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Wait for the initial query
        data = await websocket.receive_text()
        query = data
        
        # Stream the graph execution
        # We use astream to capture state changes at each node
        async for event in app_graph.astream({"query": query}, stream_mode="updates"):
            # event is a dict where keys are node names and values are the updates
            for node_name, update in event.items():
                await websocket.send_json({
                    "type": "node_update",
                    "node": node_name,
                    "update": update,
                    "status": "processing"
                })
        
        # Final result
        final_state = await app_graph.ainvoke({"query": query})
        await websocket.send_json({
            "type": "final_report",
            "report": final_state["final_report"],
            "verification": final_state["verification"],
            "status": "completed"
        })
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        print(f"Error: {e}")
