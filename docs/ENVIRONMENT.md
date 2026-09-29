# 环境与依赖配置

面向在同一台 GPU 服务器上部署、运行 **memory-online-sdk**（分层在线记忆）。  
接口说明见 [API.md](./API.md)。

---

## 1. Python 与安装

| 项 | 要求 |
|----|------|
| Python | **≥ 3.10** |
| GPU | **必须**（检索 embedding 拒绝 CPU 回退） |
| CUDA | 与所装 `torch` 版本匹配 |

```bash
cd /path/to/memory-online-sdk
python -m venv .venv && source .venv/bin/activate   # 可选
pip install -r docs/requirements.txt
pip install -e ".[test]"   # 开发/跑单测
```

也可只装运行时依赖：

```bash
pip install -r docs/requirements.txt
```

---

## 2. 配置文件

```bash
cp config/memory_online.example.json config/memory_online.json
```

`MemoryClient(config_path=...)` 或 `MemoryService` 默认读取仓库内 `config/memory_online.json`。

---

## 3. 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` | 后台抽取要 | observe / consolidate / profile（**retrieve 不调 LLM**） |
| `MEMORY_ONLINE_LLM_BASE_URL` | 否 | OpenAI 兼容 API 根 URL，默认 DeepSeek |
| `MEMORY_ONLINE_EMBEDDING_DIR` | 推荐 | 本地 embedding 模型目录（含 `config.json`），见下文 |
| `MEMORY_ONLINE_ENV_FILE` | 否 | 从此文件读 key（`KEY=value`），不覆盖已 export 的变量 |
| `MEMORY_ONLINE_LLM_CACHE` | 否 | LLM 响应磁盘缓存目录 |
| `MEMORY_ONLINE_KG_COURSE_DIR` | 否 | 课程 KG 目录（`entity_final.json` / `relations_final.json`），仅 `kp_resolver: kg:*` |

---

## 4. Embedding 模型（三层检索共用）

L0 / L1 / L2 检索均使用 **同一套** embedding 模型（默认 `BAAI/bge-large-zh-v1.5`）。

### 4.1 直接使用当前目录 / 项目下的模型（同机多人共用）

**推荐**：模型已放在服务器某路径时，在配置里写 **绝对路径**（所有人路径一致即可）：

```json
{
  "embedding_model_id": "BAAI/bge-large-zh-v1.5",
  "embedding_model_dir": "/home/shared/models/bge-large-zh-v1.5",
  "embedding_device": "cuda:0",
  "embedding_use_fp16": true
}
```

或在项目根下建 `models/`（不提交权重，仅本地或共享盘）：

```bash
# 示例：链到已有权重，避免重复下载
mkdir -p models
ln -s /home/cfh77/Documents/Chatchat/local_LLMs/bge-large-zh-v1.5 models/bge-large-zh-v1.5
```

在 **`config/memory_online.json`** 中（**启动进程时工作目录建议在项目根**，相对路径按 cwd 解析）：

```json
"embedding_model_dir": "./models/bge-large-zh-v1.5"
```

等效写法：export 环境变量（优先级见 `resolve_embedding_model_dir`）：

```bash
export MEMORY_ONLINE_EMBEDDING_DIR=/home/shared/models/bge-large-zh-v1.5
```

目录内须包含 HuggingFace 式文件，至少 **`config.json`** + 权重（如 `pytorch_model.bin` 或 `model.safetensors`）。

### 4.2 解析顺序

1. `memory_online.json` 的 **`embedding_model_dir`**（非空且含 `config.json`）  
2. 环境变量 **`MEMORY_ONLINE_EMBEDDING_DIR`**  
3. 本机 HuggingFace 缓存 `~/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5/...`

### 4.3 GPU 设备

按机器改 **`embedding_device`**，例如 `cuda:0`、`cuda:6`。多用户同机时请协调 GPU 编号，避免 OOM。

---

## 5. 抽取用文本模型（非 embedding）

在 `config/memory_online.json` 中：

```json
"extract_model": "deepseek-chat"
```

厂商与 HTTP 方式在 **`se/core/memory/online/llm.py`** 的 `_chat_completions` 修改，见 [API.md#llm-调用与替换](./API.md#llm-调用与替换)。

---

## 6. 数据与缓存（勿提交 Git）

| 路径 | 用途 |
|------|------|
| `store/` 或配置 `store_dir` | 学习者记忆持久化 |
| `.cache/memory_online_llm/` | LLM 缓存（可 `MEMORY_ONLINE_LLM_CACHE` 改） |
| `config/memory_online.json` | 本地配置（含路径；密钥请用环境变量） |

`.gitignore` 已忽略 `data/`、`store/`、`.env*`。

---

## 7. 最小自检

```bash
pip install -r docs/requirements.txt
export DEEPSEEK_API_KEY=...
export MEMORY_ONLINE_EMBEDDING_DIR=/path/to/bge-large-zh-v1.5
python -m pytest tests/test_kp_resolver.py tests/test_consolidate_validation.py -q
```

需要 GPU + 权重齐全；无 key 时单测仍可跑（不触发 live LLM）。

---

## 8. 同服务器协作说明

- **Embedding**：所有人可共用 **同一 `embedding_model_dir`**（只读挂载即可），无需每人下载一份。  
- **API Key**：每人或每服务用自己的 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`。  
- **store_dir**：按 **`learner_id`** 分目录哈希存储；不同业务线请用不同 `store_dir` 或不同部署实例，避免混写。
