# AI Customer - 智能客服系统

基于 LangGraph + Ernie 3.0 的情感分析与意图识别智能客服系统。

## 技术栈

- **LLM**: 百度文心一言（dashscope）
- **情感分析**: Ernie 3.0-base-zh 微调模型
- **流程编排**: LangGraph
- **数据库**: PostgreSQL
- **前端**: Streamlit

## 快速开始

```bash
cd text
pip install -r requirements.txt
python init_tables.py          # 初始化数据库表
streamlit run streamlit_app.py # 启动 Web 服务
```
