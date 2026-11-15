"""Measure real service latency; no generated or simulated timing values."""
import argparse
import asyncio
import json
import time
from pathlib import Path
import httpx
import numpy as np


async def measure(args):
    content=Path(args.wav).read_bytes()
    durations=[]
    async with httpx.AsyncClient(timeout=300) as client:
        # Warm up separately so lazy imports/model initialization do not hide steady-state costs.
        for index in range(args.requests+1):
            started=time.perf_counter()
            response=await client.post(args.url,files={"file":("recording.wav",content,"audio/wav")},
                headers={"X-API-Key":args.api_key} if args.api_key else {})
            response.raise_for_status()
            if index:
                durations.append(time.perf_counter()-started)
    return {"requests":len(durations),"warmup_requests":1,
        "p50_seconds":float(np.percentile(durations,50)),"p95_seconds":float(np.percentile(durations,95)),
        "mean_seconds":float(np.mean(durations)),"requests_per_second_serial":len(durations)/sum(durations)}


def main():
    import os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("--url",default="http://localhost:8002/predict")
    parser.add_argument("--requests",type=int,default=20)
    parser.add_argument("--output",default="vault/evaluations/latency.json")
    args=parser.parse_args()
    if args.requests<2: parser.error("At least two measured requests required")
    args.api_key=os.getenv("API_KEY", "")
    result=asyncio.run(measure(args))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
