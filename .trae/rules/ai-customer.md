使用langchain/langgraph框架。
修改或删除内容需经过同意。
数据库使用postgresql，使用order表查询订单信息，不使用其他表且不对表内容做任何修改，链接：dsn = "postgresql://yang:***@localhost:5432/postgres?sslmode=disable"
使用本地三分类情感模型，路径：E:\AI_Customer\sentiment_ernie_v1\checkpoint-12657
程序测试运行只尝试三次，三次失败后通知用户。
在线模型使用阿里百炼：client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
