from zhipuai import ZhipuAI

client = ZhipuAI(api_key="")  # 请填写您自己的APIKey

result = client.files.create(
    file=open("batch.jsonl", "rb"),
    purpose="batch"
)
print(result.id)
from zhipuai import ZhipuAI
client = ZhipuAI(api_key="")  
create = client.batches.create(
    input_file_id="file_123",
    endpoint="/v4/chat/completions",
    auto_delete_input_file=True,
    metadata={
        "description": "Sentiment classification"
    }
)
print(create)