import base64
import sys
from langfuse import get_client
from langfuse.openai import OpenAI  # type: ignore # drop-in wrapper: traces every call

client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")
img = base64.b64encode(open(sys.argv[1], "rb").read()).decode()

resp = client.chat.completions.create(
    model=sys.argv[2],
    temperature=0,
    seed=42,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "List every field label on this form as JSON keys. Use null for any field that is empty.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                },
            ],
        }
    ],
)
print(resp.choices[0].message.content)
get_client().flush()
