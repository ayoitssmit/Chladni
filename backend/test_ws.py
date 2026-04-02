import asyncio
import websockets
import json

async def test_connection():
    url = "ws://localhost:8000/ws/simulate?fraction=0.4&total_steps=1000"
    print(f"Connecting to {url}...")
    
    try:
        async with websockets.connect(url) as ws:
            # Receive the first 3 snapshots
            for i in range(3):
                message = await ws.recv()
                data = json.loads(message)
                print(f"\n[Snapshot {i+1}]")
                print(f"Step: {data['step']}")
                print(f"Train Acc: {data['train_accuracy']:.2%}")
                print(f"Test Acc: {data['test_accuracy']:.2%}")
                # We won't print the whole PCA array to keep the terminal clean
                print(f"PCA Embeddings received: {len(data['pca_embeddings'])} points")
                
            print("\nConnection test successful!")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
