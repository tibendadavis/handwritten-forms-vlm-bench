import argparse
import base64
import mimetypes
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: one image, one traced VLM call.")
    parser.add_argument("image", type=Path, help="path to a form photo")
    parser.add_argument("model", help="Ollama model tag, e.g. qwen2.5vl:3b")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))

    from langfuse import get_client
    from langfuse.openai import OpenAI  # type: ignore # drop-in wrapper: traces every call

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="local")

    img = base64.b64encode(args.image.read_bytes()).decode()
    mime = mimetypes.guess_type(args.image)[0] or "image/jpeg"

    resp = client.chat.completions.create(
        model=args.model,
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


if __name__ == "__main__":
    main()
