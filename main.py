from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from starlette.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware

# 初始化 FastAPI
app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，或者指定具体域名如 ["http://localhost"]
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 加载模型和分词器
model_path = "LLM4Binary/llm4decompile-6.7b-v1.5"  # 替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()

# 请求体格式
class DecompileRequest(BaseModel):
    asm_code: str

# 反编译函数
def decompile(asm_code: str) -> str:
    prompt = f"# This is the assembly code:\n{asm_code.strip()}\n# What is the source code?\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    return tokenizer.decode(outputs[0][len(inputs[0]):-1], skip_special_tokens=True)

# API 端点
@app.post("/decompile")
async def decompile_endpoint(request: DecompileRequest):
    try:
        result = decompile(request.asm_code)
        return {"source_code": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# 主页 - 提供静态index.html
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)