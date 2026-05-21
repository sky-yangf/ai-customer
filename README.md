# AI Customer - 智能客服系统

基于 LangGraph + Ernie 3.0 的情感分析与意图识别智能客服系统，支持人工介入。

## 系统架构

```
用户输入
    ↓
┌─────────────────┐
│  情感分析节点   │ ← Ernie 3.0 微调模型（三分类：消极/中性/积极）
└────────┬────────┘
         ↓
┌─────────────────┐
│  意图识别节点   │ ← 阿里百炼 Qwen-Turbo
│  1=查询订单     │
│  2=查询产品     │
│  3=要求退货     │
│  4=转人工       │
│  5=闲聊         │
└────────┬────────┘
         ↓
    ┌────┴────┐
    │  分支路由  │
    └────┬────┘
         ↓
    ┌────┼────┬────┬────┐
    ↓    ↓    ↓    ↓    ↓
  订单  产品  退货  人工  闲聊
  查询  咨询  处理  介入  回复
    └────┬────┬────┬────┘
         ↓
┌─────────────────┐
│  响应组装节点   │
└────────┬────────┘
         ↓
      用户回复
```

## 功能特性

- **情感分析**：基于本地微调 Ernie 3.0 模型，消极情绪触发安抚话术
- **意图识别**：5 类意图自动分流（订单查询/产品咨询/退货处理/转人工/闲聊）
- **人工介入**：客服工作台实时接收待处理工单，支持上下翻页查看历史
- **工单管理**：待处理/已处理工单分类，urgency 紧急程度评估
- **对话历史**：用户端支持历史对话切换与删除

## 目录结构

```
AI_Customer/
├── text/
│   ├── langgraph_flow.py      # 核心 LangGraph 流程编排
│   ├── streamlit_app.py       # 客户端 Web 界面（用户端）
│   ├── streamlit_agent.py     # 客服工作台（客服端）
│   ├── init_tables.py         # 数据库表初始化
│   ├── create_agent.py        # 创建客服账号
│   ├── delete_agent.py        # 删除客服账号
│   └── train_sentiment.py     # 情感分析模型训练脚本
├── .trae/rules/
│   └── ai-customer.md         # Trae IDE 开发规则
├── sentiment_ernie_v1/        # Ernie 3.0 微调模型（需单独下载）
│   └── checkpoint-12657/
├── models/                    # 基座模型 & 数据集（需单独下载）
│   ├── nghuyong--ernie-3.0-base-zh/
│   └── datasets/
│       └── balanced_data.csv
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- PostgreSQL 14+
- CUDA（可选，用于加速情感分析推理）

### 2. 安装依赖

```bash
cd text
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
export DASHSCOPE_API_KEY="your-api-key"   # 阿里百炼 API Key
```

### 4. 初始化数据库

```bash
# 创建 order 表（存放订单数据）
psql -U your_user -d postgres -c "
CREATE TABLE IF NOT EXISTS \"order\" (
    order_no VARCHAR(64) PRIMARY KEY,
    status VARCHAR(32),
    product_name VARCHAR(256)
);"

# 初始化对话 & 工单表
python init_tables.py
```

### 5. 下载模型权重

情感分析模型（sentiment_ernie_v1）需单独获取，放到项目根目录：
- 百度网盘 / HuggingFace（待上传）

### 6. 启动服务

**客户端界面（用户端）**
```bash
streamlit run text/streamlit_app.py --server.port 8501
```

**客服工作台**
```bash
streamlit run text/streamlit_agent.py --server.port 8502
```

## 数据库表结构

| 表名 | 说明 |
|------|------|
| `order` | 订单信息（需提前导入） |
| `conversation_history` | 用户对话历史 |
| `human_intervention` | 人工介入工单 |
| `agent_replies` | 客服回复记录 |
| `users` | 客户端用户账号 |
| `agents` | 客服账号 |

## 默认账号

| 类型 | 账号 | 密码 |
|------|------|------|
| 客服 | kefu01 | 123456 |

（通过 `create_agent.py` 创建，初始无可用客服账号）

## 开发规范

详见 [`.trae/rules/ai-customer.md`](.trae/rules/ai-customer.md)

- LangGraph 框架
- PostgreSQL 数据库（仅读 `order` 表）
- 本地 Ernie 3.0 微调模型
- 阿里百炼 API
