# memory-online-sdk

分层在线记忆 SDK（L0 观察 → L1 知识点卡 → L2 特质）。**不生成辅导回复**；业务方自行调用 LLM 讲题，本库负责检索与写入。

## 快速开始

```bash
pip install -e ".[test]"
cp config/memory_online.example.json config/memory_online.json
export DEEPSEEK_API_KEY=...
export MEMORY_ONLINE_EMBEDDING_DIR=/path/to/bge-large-zh-v1.5
```

```python
from se.core.memory.online.api import MemoryClient

client = MemoryClient(store_dir="./store")
r = client.retrieve("learner_1", "sess_1", "我还是不懂通分", question_text="1/2 + 1/3")
# 你们的辅导 LLM：system + r.prompt_block + 学生话 → reply
client.commit_turn("learner_1", "sess_1", "我还是不懂通分", reply, msg_id="m-001")
client.end_session("learner_1", "sess_1")
client.wait_pending()
client.close()
```

接口说明见 [docs/API.md](docs/API.md)。  
**修改 LLM 调用方式**：见 [docs/API.md#llm-调用与替换](docs/API.md#llm-调用与替换)。

## 范围

本仓库仅包含上线所需 `se/core/memory/online` 及最小依赖，不含评测脚本与数据集。
