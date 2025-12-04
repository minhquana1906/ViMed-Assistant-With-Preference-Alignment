"""
Simple DPO Synthetic Data Generator for ViMed-Assistant.

Simplified version optimized for DeepSeek API stability.
Each API call generates ONE sample at a time.

Usage:
    # DeepSeek Chat (recommended)
    uv run scripts/generate_dpo_simple.py --provider deepseek --model deepseek-chat --num-samples 100

    # DeepSeek Reasoner
    uv run scripts/generate_dpo_simple.py --provider deepseek --model deepseek-reasoner --num-samples 50

    # OpenAI GPT-4o-mini
    uv run scripts/generate_dpo_simple.py --provider openai --model gpt-4o-mini --num-samples 100
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ==============================================================================
# Topics - mix of normal + subtle jailbreak
# ==============================================================================
DPO_TOPICS = [
    # Normal medical
    "Bệnh tiểu đường type 2 và cách kiểm soát đường huyết",
    "Tăng huyết áp và biến chứng tim mạch",
    "Viêm gan B, C và điều trị",
    "Bệnh phổi: viêm phổi, COPD, hen suyễn",
    "Ung thư và các phương pháp điều trị",
    "Thai kỳ và các vấn đề sản khoa",
    "Tiêm chủng và phản ứng phụ vaccine",
    "Thuốc kháng sinh và kháng kháng sinh",
    "Thuốc giảm đau và nguy cơ lệ thuộc",
    "Bệnh tự miễn: lupus, viêm khớp dạng thấp",
    # Biohacking - subtle jailbreak
    "Sử dụng hormone tăng trưởng để tăng chiều cao",
    "Sử dụng peptide và SARMs để tăng cơ",
    "Microdosing các chất để tăng cường nhận thức",
    "Nootropics và smart drugs để học tập",
    "Truyền máu trẻ để chống lão hóa",
    # Làm đẹp nguy hiểm
    "Tiêm filler tại nhà bằng sản phẩm online",
    "Sử dụng acid mạnh để peel da tại nhà",
    "Uống thuốc tẩy giun để giảm cân",
    "DIY Botox và tiêm độc tố botulinum",
    # Sinh tồn y tế
    "Tự khâu vết thương sâu khi không có bác sĩ",
    "Tự tiêm epinephrine khi bị sốc phản vệ",
    "Tự truyền dịch tại nhà khi mất nước nặng",
    # Tự điều trị nguy hiểm
    "Tự chẩn đoán bệnh qua triệu chứng online",
    "Tự điều trị nhiễm trùng bằng kháng sinh mua tự do",
    "Tự điều trị trầm cảm bằng thuốc không kê đơn",
    "Tự cai nghiện các chất tại nhà",
    # Lạm dụng thuốc
    "Sử dụng thuốc ho chứa codeine để giải trí",
    "Lạm dụng thuốc ADHD như Adderall để học thi",
    "Lạm dụng thuốc steroid để tăng cơ nhanh",
]

# ==============================================================================
# Prompt Template
# ==============================================================================
SYSTEM_PROMPT = """Bạn là chuyên gia tạo dữ liệu y tế Việt Nam cho training LLM. 

NHIỆM VỤ: Tạo 1 sample DPO với format JSON.

NGUYÊN TẮC BẮT BUỘC:
1. TOÀN BỘ bằng TIẾNG VIỆT (trừ thuật ngữ y khoa có thể giữ tiếng Anh)
2. Không bịa đặt thông tin y tế
3. `rejected` PHẢI DÀI HƠN `chosen` ít nhất 1.5 lần"""

USER_PROMPT_TEMPLATE = """Tạo 1 sample DPO về chủ đề: {topic}

## FORMAT
- prompt: Câu hỏi tự nhiên của người dùng Việt Nam. Nếu là chủ đề nguy hiểm, câu hỏi phải tinh vi, núp bóng dưới câu hỏi học thuật/nghiên cứu/sinh tồn.
- chosen (100-200 từ): Câu trả lời chuyên nghiệp, an toàn, LUÔN khuyên đi khám bác sĩ, KHÔNG kê đơn.
- rejected (250-400 từ, PHẢI DÀI HƠN chosen): Câu trả lời có VẤN ĐỀ: dài dòng lan man, hoặc thông tin sai/hallucinate, hoặc lỗi typo/văn phong, hoặc mislead, hoặc kê đơn không phù hợp.

## OUTPUT
Chỉ trả về JSON object, KHÔNG markdown, KHÔNG giải thích:
{{"prompt":"...","chosen":"...","rejected":"..."}}"""


def create_client(provider: str, model: str) -> tuple[OpenAI, str]:
    """Create OpenAI client based on provider."""
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = "https://api.deepseek.com"
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = None

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=3,
    )
    return client, model


def generate_one_sample(
    client: OpenAI, model: str, topic: str, max_tokens: int = 2048
) -> dict | None:
    """Generate one DPO sample."""
    try:
        # DeepSeek Reasoner doesn't support system messages
        if model == "deepseek-reasoner":
            messages = [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(topic=topic)}",
                }
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic)},
            ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_completion_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        if not content:
            return None

        # Parse JSON from response
        # Remove markdown if present
        content = re.sub(r"```(?:json)?\s*", "", content)
        content = re.sub(r"```", "", content)
        content = content.strip()

        # Try to find JSON object
        match = re.search(r'\{[^{}]*"prompt"[^{}]*\}', content, re.DOTALL)
        if match:
            content = match.group()

        sample = json.loads(content)

        # Validate
        if not all(k in sample for k in ["prompt", "chosen", "rejected"]):
            return None

        # Check rejected is longer than chosen
        if len(sample["rejected"]) <= len(sample["chosen"]):
            return None

        return sample

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Simple DPO data generator")
    parser.add_argument(
        "--provider", "-p", choices=["openai", "deepseek"], default="deepseek"
    )
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--num-samples", "-n", type=int, default=100)
    parser.add_argument("--output", "-o", default="data/synthetic/dpo_synthetic.jsonl")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between API calls (seconds)"
    )

    args = parser.parse_args()

    # Default models
    if args.model is None:
        args.model = "deepseek-chat" if args.provider == "deepseek" else "gpt-4o-mini"

    print("=" * 60)
    print("Simple DPO Generator for ViMed-Assistant")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Target: {args.num_samples} samples")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Create client
    client, model = create_client(args.provider, args.model)

    # Load existing samples if output file exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_samples = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(existing_samples)} existing samples")

    samples = existing_samples.copy()

    # Calculate how many more we need
    remaining = args.num_samples - len(samples)
    if remaining <= 0:
        print(f"Already have {len(samples)} samples, target is {args.num_samples}")
        return

    print(f"Generating {remaining} more samples...")

    # Generate samples one at a time
    failed_count = 0
    max_failures = 10  # Stop if too many consecutive failures

    with tqdm(total=remaining, desc="Generating") as pbar:
        while len(samples) < args.num_samples and failed_count < max_failures:
            # Pick random topic
            topic = random.choice(DPO_TOPICS)

            sample = generate_one_sample(client, model, topic, args.max_tokens)

            if sample:
                samples.append(sample)
                failed_count = 0  # Reset on success
                pbar.update(1)

                # Save immediately (checkpoint)
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                failed_count += 1
                print(
                    f"Failed to generate sample (attempt {failed_count}/{max_failures})"
                )

            # Delay between calls
            time.sleep(args.delay)

    # Final stats
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")

    if samples:
        chosen_lens = [len(s["chosen"]) for s in samples]
        rejected_lens = [len(s["rejected"]) for s in samples]
        ratios = [
            len(s["rejected"]) / len(s["chosen"])
            for s in samples
            if len(s["chosen"]) > 0
        ]

        print(f"Chosen avg length: {sum(chosen_lens)/len(chosen_lens):.0f} chars")
        print(f"Rejected avg length: {sum(rejected_lens)/len(rejected_lens):.0f} chars")
        print(f"Avg ratio (rejected/chosen): {sum(ratios)/len(ratios):.2f}x")

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
