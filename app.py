import os
import uuid
import tempfile
import subprocess
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import faiss

app = FastAPI(title="SpeakerID Service (MVP)")

encoder = VoiceEncoder()

STORE = {}
EMB_DIM = 256
DATA_DIR = "/tmp/speakerid_data"
os.makedirs(DATA_DIR, exist_ok=True)

class EnrollReq(BaseModel):
    tenant_id: str
    name: str
    audio_url: str

class IdentifyReq(BaseModel):
    tenant_id: str
    audio_url: str
    start: Optional[float] = None
    end: Optional[float] = None

def download_file(url, outpath):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(outpath, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return outpath

def cut_audio_with_ffmpeg(src_path, out_path, start=None, end=None):
    cmd = ["ffmpeg", "-y", "-i", src_path]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-ar", "16000", "-ac", "1", out_path]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def ensure_index(tenant_id):
    if tenant_id not in STORE:
        idx = faiss.IndexFlatIP(EMB_DIM)
        meta = []
        STORE[tenant_id] = {"index": idx, "meta": meta}
    return STORE[tenant_id]

@app.post("/enroll")
def enroll(req: EnrollReq):
    tmp_src = os.path.join(DATA_DIR, f"enroll_{uuid.uuid4()}.wav")
    download_file(req.audio_url, tmp_src)
    wav = preprocess_wav(tmp_src)
    emb = encoder.embed_utterance(wav)
    emb = emb / np.linalg.norm(emb)
    store = ensure_index(req.tenant_id)
    idx = store["index"]
    meta = store["meta"]
    idx.add(np.expand_dims(emb, axis=0).astype("float32"))
    profile_id = str(uuid.uuid4())
    meta.append({"profile_id": profile_id, "name": req.name})
    return {"status":"ok", "profile_id": profile_id, "name": req.name}

@app.post("/identify")
def identify(req: IdentifyReq):
    tmp_src = os.path.join(DATA_DIR, f"src_{uuid.uuid4()}.wav")
    try:
        download_file(req.audio_url, tmp_src)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"download error: {e}")
    tmp_seg = os.path.join(DATA_DIR, f"seg_{uuid.uuid4()}.wav")
    try:
        if req.start is not None or req.end is not None:
            cut_audio_with_ffmpeg(tmp_src, tmp_seg, start=req.start, end=req.end)
            use_path = tmp_seg
        else:
            cut_audio_with_ffmpeg(tmp_src, tmp_seg, start=None, end=None)
            use_path = tmp_seg
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="ffmpeg cut failed")
    wav = preprocess_wav(use_path)
    emb = encoder.embed_utterance(wav)
    emb = emb / np.linalg.norm(emb)
    if req.tenant_id not in STORE:
        return {"name":"Unknown","score":0.0}
    store = STORE[req.tenant_id]
    idx = store["index"]
    meta = store["meta"]
    k = min(3, idx.ntotal) if idx.ntotal>0 else 0
    if k == 0:
        return {"name":"Unknown","score":0.0}
    D, I = idx.search(np.expand_dims(emb, axis=0).astype("float32"), k)
    sims = D[0].tolist()
    ids = I[0].tolist()
    top_score = float(sims[0])
    top_idx = ids[0]
    top_meta = meta[top_idx]
    second = float(sims[1]) if len(sims)>1 else 0.0
    ambiguous = (top_score - second) < 0.05
    if top_score < 0.75:
        return {"name":"Unknown","score":top_score,"ambiguous":ambiguous}
    return {"name": top_meta["name"], "profile_id": top_meta["profile_id"], "score": top_score, "ambiguous": ambiguous}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
