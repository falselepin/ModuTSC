# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modutsc.scheduling.registry import discover, discover_user_plugins
from modutsc.api import invalidate_compatibility_cache

discover("modutsc")  # 预先加载所有内置模块，确保注册生效
discover_user_plugins()  # 加载用户上传的自定义插件
invalidate_compatibility_cache()  # 重建兼容性缓存（包含用户插件）


# 导入各路由模块
from app.routers import datasets
from app.routers import models
from app.routers import compare
from app.routers import assemble
from app.routers import experiments as experiments_router   # 避免与变量名冲突
from app.routers import snapshot
from app.routers import configs
from app.routers import agent
from app.routers import plugins

# 导入文件服务
from app.services.file_service import load_index
from app.services.experiment_service import experiments   # 这是全局实验列表

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 挂载路由
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(compare.router)
app.include_router(assemble.router)
app.include_router(experiments_router.router)   # 注意这里用别名
app.include_router(snapshot.router)
app.include_router(configs.router)
app.include_router(agent.router)
app.include_router(plugins.router)

@app.on_event("startup")
def startup():
    experiments.clear()
    experiments.extend(load_index())