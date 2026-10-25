import base64
import mimetypes
import sys
from pathlib import Path

from langfuse import get_client
from langfuse.openai import OpenAI  # type: ignore # drop-in wrapper: traces every call

image_path = Path(sys.argv[1])
model = sys.argv[2]

client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")

img = base64.b64encode(image_path.read_bytes()).decode()
mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"

resp = client.chat.completions.create(
    model=model,
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
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img}"}},
            ],
        }
    ],
)
print(resp.choices[0].message.content)
get_client().flush()
