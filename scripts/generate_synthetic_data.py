"""
Usage:
    uv run scripts/generate_synthetic_data.py \
    --provider deepseek \
    --num-samples 100 \
    --batch-size 5 \
    --max-concurrent 20 \
    --max-tokens 1536 \
    --temperature 1.0
    
    uv run scripts/generate_synthetic_data.py \
    --provider openai \
    --num-samples 10 \
    --batch-size 5 \
    --max-concurrent 20 \
    --max-tokens 1536 \
    --model gpt-4o-mini 
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("synthetic_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class GenerationConfig:

    provider: Literal["openai", "deepseek"] = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = None
    base_url: str | None = None

    batch_size: int = 5  # Q&A pairs per API call (optimal: 5-10)
    max_concurrent_requests: int = 15  # Concurrent API requests
    temperature: float = 1.0
    max_retries: int = 3

    # Control output length (Hard limit)
    max_tokens: int = 1536

    # Use sync mode for stability (slower but reliable)
    sync_mode: bool = False

    num_samples_to_generate: int = 1000
    output_file: str = "data/synthetic/sft_synthetic.jsonl"
    checkpoint_file: str = "tmp/generation_checkpoint_gpt.jsonl"

    def __post_init__(self):
        if self.provider == "deepseek":
            self.base_url = "https://api.deepseek.com"
            if not self.api_key:
                self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        elif self.provider == "openai":
            self.model = self.model or "gpt-4o-mini"
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY", "")


# ==============================================================================
# Medical Topics
# ==============================================================================
MEDICAL_TOPICS = [
    # Nội khoa
    ("Nội khoa", "Bệnh tiểu đường type 1, type 2"),
    ("Nội khoa", "Tăng huyết áp và biến chứng"),
    ("Nội khoa", "Bệnh tim mạch, suy tim, nhồi máu cơ tim"),
    ("Nội khoa", "Bệnh phổi: viêm phổi, COPD, hen suyễn"),
    ("Nội khoa", "Bệnh gan: viêm gan B, C, xơ gan"),
    ("Nội khoa", "Bệnh thận: suy thận, sỏi thận"),
    ("Nội khoa", "Bệnh dạ dày - ruột: trào ngược, viêm loét"),
    ("Nội khoa", "Rối loạn lipid máu"),
    ("Nội khoa", "Thiếu máu các loại"),
    ("Nội khoa", "Bệnh tuyến giáp"),
    # Ngoại khoa
    ("Ngoại khoa", "Viêm ruột thừa cấp"),
    ("Ngoại khoa", "Sỏi mật và viêm túi mật"),
    ("Ngoại khoa", "Thoát vị bẹn, thoát vị đĩa đệm"),
    ("Ngoại khoa", "U lành và u ác tính"),
    ("Ngoại khoa", "Chấn thương và gãy xương"),
    ("Ngoại khoa", "Bỏng và xử trí vết thương"),
    # Sản phụ khoa
    ("Sản phụ khoa", "Thai kỳ và các vấn đề liên quan"),
    ("Sản phụ khoa", "Các bệnh phụ khoa thường gặp"),
    ("Sản phụ khoa", "U xơ tử cung, u nang buồng trứng"),
    ("Sản phụ khoa", "Rối loạn kinh nguyệt"),
    ("Sản phụ khoa", "Mãn kinh và triệu chứng"),
    # Nhi khoa
    ("Nhi khoa", "Bệnh nhiễm trùng thường gặp ở trẻ"),
    ("Nhi khoa", "Tiêm chủng và lịch tiêm phòng"),
    ("Nhi khoa", "Dinh dưỡng và phát triển của trẻ"),
    ("Nhi khoa", "Bệnh đường hô hấp ở trẻ"),
    ("Nhi khoa", "Tiêu chảy và mất nước ở trẻ"),
    ("Nhi khoa", "Sốt xuất huyết ở trẻ em"),
    # Da liễu
    ("Da liễu", "Viêm da cơ địa, eczema"),
    ("Da liễu", "Mụn trứng cá và các loại mụn"),
    ("Da liễu", "Nấm da, nấm móng"),
    ("Da liễu", "Bệnh vẩy nến"),
    ("Da liễu", "Dị ứng da và mề đay"),
    # Nhãn khoa
    ("Nhãn khoa", "Cận thị, viễn thị, loạn thị"),
    ("Nhãn khoa", "Đục thủy tinh thể"),
    ("Nhãn khoa", "Glaucoma (Tăng nhãn áp)"),
    ("Nhãn khoa", "Viêm kết mạc, khô mắt"),
    # Tai Mũi Họng
    ("Tai Mũi Họng", "Viêm họng, viêm amidan"),
    ("Tai Mũi Họng", "Viêm xoang cấp và mạn"),
    ("Tai Mũi Họng", "Viêm tai giữa"),
    ("Tai Mũi Họng", "Rối loạn tiền đình"),
    # Cơ xương khớp
    ("Cơ xương khớp", "Thoái hóa khớp"),
    ("Cơ xương khớp", "Viêm khớp dạng thấp"),
    ("Cơ xương khớp", "Loãng xương"),
    ("Cơ xương khớp", "Đau lưng và thoát vị đĩa đệm"),
    ("Cơ xương khớp", "Gout (Gút)"),
    # Sức khỏe tâm thần
    ("Sức khỏe tâm thần", "Trầm cảm và lo âu"),
    ("Sức khỏe tâm thần", "Rối loạn giấc ngủ"),
    ("Sức khỏe tâm thần", "Stress và cách quản lý"),
    # Dinh dưỡng
    ("Dinh dưỡng", "Chế độ ăn cho người tiểu đường"),
    ("Dinh dưỡng", "Chế độ ăn cho người tăng huyết áp"),
    ("Dinh dưỡng", "Dinh dưỡng cho bà bầu"),
    ("Dinh dưỡng", "Dinh dưỡng cho trẻ em"),
]

SYSTEM_PROMPT = """Bạn là một trợ lý y tế ảo thông minh, với vai trò là một bác sĩ tư vấn trực tuyến chuyên nghiệp và tận tâm. Nhiệm vụ của bạn là giải đáp thắc mắc, câu hỏi về chủ đề y tế. Câu trả lời cần mang tính định hướng, giải thích nguyên nhân có thể, không được thay thế chẩn đoán của bệnh viện và phải luôn khuyên người dùng đến cơ sở y tế để có chẩn đoán chính xác."""

GENERATION_PROMPT = """Tạo {num_pairs} cặp câu hỏi-trả lời y tế về: {topic} - {subtopic}

## YÊU CẦU
- Độ dài trả lời: từ 400 đến 1024 từ (hoặc tương đương 350 đến 700 tokens)
- Câu hỏi tự nhiên như bệnh nhân thực sự hỏi
- Câu trả lời chính xác y khoa, LUÔN khuyên đi khám bác sĩ
- KHÔNG kê đơn thuốc cụ thể, KHÔNG chẩn đoán xác định
- Dùng markdown format, headings, bullet points khi cần
- Tiếng Việt, tên thuốc có thể tiếng Anh
- Chỉ trả về JSON array, KHÔNG giải thích thêm, không đưa vào code block, markdown

## OUTPUT (JSON array)
[{{"question": "...", "answer": "..."}}]"""


# ==============================================================================
# Basic Guardrails
# ==============================================================================
HARMFUL_PATTERNS = [
    r"tự\s*tử|tự\s*sát",
    r"phá\s*thai\s*tại\s*nhà",
    r"liều.*\d+\s*mg.*ngày",
    r"chắc\s*chắn\s*bạn\s*bị",
    r"không\s*cần.*đi\s*khám",
]

REQUIRED_PATTERNS = [
    r"(?:nên|cần|hãy).*(?:đi\s*khám|khám|bác\s*sĩ|cơ\s*sở\s*y\s*tế|bệnh\s*viện|chuyên\s*khoa)",
    r"tư\s*vấn.*(?:bác\s*sĩ|chuyên\s*gia|y\s*tế)",
    r"(?:chẩn\s*đoán|thăm\s*khám|kiểm\s*tra).*(?:chính\s*xác|cụ\s*thể|kỹ)",
    r"(?:đến|tới).*(?:bệnh\s*viện|phòng\s*khám|cơ\s*sở)",
    r"(?:bác\s*sĩ|chuyên\s*gia).*(?:tư\s*vấn|khám|chẩn\s*đoán)",
]


def quick_validate(answer: str) -> bool:
    if len(answer) < 30 or len(answer) > 5000:
        return False

    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return False

    for pattern in REQUIRED_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return True

    if len(answer) > 200:
        return True

    return False


# ==============================================================================
# Generator
# ==============================================================================
class FastSyntheticGenerator:

    def __init__(self, config: GenerationConfig):
        self.config = config

        if config.sync_mode:
            self.sync_client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=120.0,
                max_retries=config.max_retries,
            )
        else:
            self.async_client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=120.0,
                max_retries=0,
            )

        self.generated = 0
        self.accepted = 0
        self.rejected = 0

    def _call_api_sync(self, prompt: str) -> list[dict]:
        try:
            response = self.sync_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Generate Vietnamese medical Q&A data in JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
            return self._parse_result(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Sync API error: {type(e).__name__}: {e}")
            return []

    async def _call_api_async(self, prompt: str) -> list[dict]:
        for attempt in range(self.config.max_retries):
            try:
                effective_max_tokens = self.config.max_tokens
                if self.config.model == "gpt-4o-mini" and self.config.max_tokens < 4096:
                    effective_max_tokens = max(4096, self.config.batch_size * 1500)

                if self.config.model == "gpt-4o-mini":
                    response = await asyncio.wait_for(
                        self.async_client.chat.completions.create(
                            model=self.config.model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Generate Vietnamese medical Q&A data in JSON format.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=self.config.temperature,
                            max_tokens=effective_max_tokens,
                        ),
                        timeout=120.0,
                    )
                else:
                    response = await asyncio.wait_for(
                        self.async_client.chat.completions.create(
                            model=self.config.model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Generate Vietnamese medical Q&A data in JSON format.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            response_format={"type": "json_object"},
                        ),
                        timeout=120.0,
                    )

                content = response.choices[0].message.content
                if not content or content.strip() == "":
                    logger.warning(
                        f"Empty response content. Finish reason: {response.choices[0].finish_reason}"
                    )
                    if (
                        hasattr(response.choices[0].message, "refusal")
                        and response.choices[0].message.refusal
                    ):
                        logger.warning(
                            f"Model refusal: {response.choices[0].message.refusal}"
                        )
                    continue

                return self._parse_result(content)

            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout on attempt {attempt+1}/{self.config.max_retries}"
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {type(e).__name__}: {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
        return []

    def _parse_result(self, result_text: str) -> list[dict]:
        try:
            result = json.loads(result_text)

            if isinstance(result, list):
                return result

            if isinstance(result, dict):
                # Try explicit keys first
                for key in [
                    "data",
                    "pairs",
                    "qa_pairs",
                    "questions",
                    "qa",
                    "items",
                    "results",
                    "qa_list",
                ]:
                    if key in result and isinstance(result[key], list):
                        return result[key]

                # If it's a single Q&A object
                if "question" in result and "answer" in result:
                    return [result]

                # Try to find ANY key that contains an array of dicts with question/answer
                for key, value in result.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict) and (
                            "question" in value[0] or "answer" in value[0]
                        ):
                            logger.info(f"Found Q&A data under key: '{key}'")
                            return value

                logger.warning(
                    f"JSON parsed but no Q&A found. Keys: {list(result.keys())}"
                )
                logger.debug(f"Response preview: {result_text[:500]}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result_text)
            if json_match:
                try:
                    extracted = json.loads(json_match.group(1))
                    logger.info("Extracted JSON from code block")
                    return self._parse_result(json.dumps(extracted))
                except json.JSONDecodeError:
                    pass
        return []

    def format_sample(self, question: str, answer: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }

    def _prepare_batches(self, processed_batches: set) -> list[tuple]:
        batch_size = self.config.batch_size
        total_topics = len(MEDICAL_TOPICS)
        total_batches_needed = max(1, self.config.num_samples_to_generate // batch_size)
        batches_per_topic = max(1, total_batches_needed // total_topics)
        extra_batches = total_batches_needed % total_topics

        batch_tasks = []
        for idx, (topic, subtopic) in enumerate(MEDICAL_TOPICS):
            topic_batches = batches_per_topic + (1 if idx < extra_batches else 0)
            for batch_idx in range(topic_batches):
                batch_id = f"{topic}:{subtopic}:{batch_idx}"
                if batch_id not in processed_batches:
                    batch_tasks.append((topic, subtopic, batch_idx, batch_id))

        logger.info(
            f"Config: batch_size={batch_size}, total_batches={total_batches_needed}, batches_per_topic≈{batches_per_topic}"
        )
        logger.info(f"Total batches to process: {len(batch_tasks)}")
        return batch_tasks

    def _process_qa_pairs(self, qa_pairs: list[dict]) -> list[dict]:
        results = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            if question and answer and quick_validate(answer):
                results.append(self.format_sample(question, answer))
                self.accepted += 1
            elif question and answer:
                self.rejected += 1
        self.generated += len(qa_pairs)
        return results

    def _load_checkpoint(self, checkpoint_path: Path) -> list:
        all_samples = []
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "messages" in data:
                            all_samples.append(data)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Resumed: {len(all_samples)} samples from checkpoint")
        return all_samples

    def _save_output(
        self, output_path: Path, checkpoint_path: Path, all_samples: list
    ) -> None:
        """Save final output and print stats."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        logger.info("=" * 50)
        logger.info(f"Generated: {self.generated}")
        logger.info(f"Accepted: {self.accepted}")
        logger.info(f"Rejected: {self.rejected}")
        logger.info(f"Final: {len(all_samples)} samples")
        logger.info(f"Output: {output_path}")

    def run_sync(self) -> None:
        output_path = Path(self.config.output_file)
        checkpoint_path = Path(self.config.checkpoint_file)

        all_samples = self._load_checkpoint(checkpoint_path)

        samples_needed = self.config.num_samples_to_generate - len(all_samples)
        if samples_needed <= 0:
            logger.info(
                f"Already have {len(all_samples)} samples, target is {self.config.num_samples_to_generate}"
            )
            self._save_output(output_path, checkpoint_path, all_samples)
            return

        batch_tasks = self._prepare_batches(set())

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_f:
            for topic, subtopic, batch_idx, batch_id in tqdm(
                batch_tasks, desc="Generating"
            ):
                prompt = GENERATION_PROMPT.format(
                    num_pairs=self.config.batch_size,
                    topic=topic,
                    subtopic=subtopic,
                )
                qa_pairs = self._call_api_sync(prompt)
                results = self._process_qa_pairs(qa_pairs)
                all_samples.extend(results)

                for sample in results:
                    checkpoint_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                checkpoint_f.flush()

                if len(all_samples) >= self.config.num_samples_to_generate:
                    logger.info(f"Reached target: {len(all_samples)} samples")
                    break

                time.sleep(0.3)

        self._save_output(output_path, checkpoint_path, all_samples)

    async def run_async(self) -> None:
        output_path = Path(self.config.output_file)
        checkpoint_path = Path(self.config.checkpoint_file)

        all_samples = self._load_checkpoint(checkpoint_path)

        samples_needed = self.config.num_samples_to_generate - len(all_samples)
        if samples_needed <= 0:
            logger.info(
                f"Already have {len(all_samples)} samples, target is {self.config.num_samples_to_generate}"
            )
            self._save_output(output_path, checkpoint_path, all_samples)
            return

        batches_needed = max(
            1, (samples_needed + self.config.batch_size - 1) // self.config.batch_size
        )
        batch_tasks = self._prepare_batches(set())[:batches_needed]

        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def process_batch(
            topic: str, subtopic: str, batch_idx: int, batch_id: str
        ):
            async with semaphore:
                prompt = GENERATION_PROMPT.format(
                    num_pairs=self.config.batch_size,
                    topic=topic,
                    subtopic=subtopic,
                )
                qa_pairs = await self._call_api_async(prompt)
                return batch_id, self._process_qa_pairs(qa_pairs)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tasks = [process_batch(t, s, i, bid) for t, s, i, bid in batch_tasks]

        with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_f:
            for future in tqdm_asyncio.as_completed(tasks, desc="Generating (Async)"):
                batch_id, samples = await future

                if not samples:
                    continue

                all_samples.extend(samples)

                for sample in samples:
                    checkpoint_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                checkpoint_f.flush()  # Force write to disk

        self._save_output(output_path, checkpoint_path, all_samples)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical Q&A data")
    parser.add_argument("--output", "-o", default="data/synthetic/sft_synthetic.jsonl")
    parser.add_argument("--num-samples", "-n", type=int, default=1000)
    parser.add_argument(
        "--provider", "-p", choices=["openai", "deepseek"], default="deepseek"
    )
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=5)
    parser.add_argument("--max-concurrent", "-c", type=int, default=10)
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1536,
        help="Max output tokens per request (Length Control)",
    )
    parser.add_argument(
        "--sync-mode",
        "-s",
        action="store_true",
        help="Use synchronous mode (slower but more stable)",
    )

    args = parser.parse_args()

    config = GenerationConfig(
        provider=args.provider,
        model=args.model
        or ("deepseek-chat" if args.provider == "deepseek" else "gpt-4o-mini"),
        batch_size=args.batch_size,
        max_concurrent_requests=args.max_concurrent,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples_to_generate=args.num_samples,
        output_file=args.output,
        sync_mode=args.sync_mode,
    )

    generator = FastSyntheticGenerator(config)

    if config.sync_mode:
        logger.info("Running in SYNC mode (stable but slower)")
        generator.run_sync()
    else:
        logger.info("Running in ASYNC mode (faster, realtime checkpointing)")
        asyncio.run(generator.run_async())


if __name__ == "__main__":
    main()
