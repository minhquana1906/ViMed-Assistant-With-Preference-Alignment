"""
Usage:
    uv run scripts/generate_synthetic_dpo_data.py \
        --provider openai \
        --model gpt-4o-mini \
        --num-samples 500 \
        --batch-size 5 \
        --max-concurrent 20
    
    uv run scripts/generate_synthetic_dpo_data.py \
        --provider deepseek \
        --model deepseek-chat \
        --num-samples 500 \
        --batch-size 5 \
        --max-concurrent 20
    
    uv run scripts/generate_synthetic_dpo_data.py \
        --provider deepseek \
        --model deepseek-reasoner \
        --num-samples 500 \
        --batch-size 3 \
        --sync-mode
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
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
        logging.FileHandler("synthetic_dpo_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class DPOGenerationConfig:
    """Configuration for DPO synthetic data generation."""

    provider: Literal["openai", "deepseek"] = "deepseek"
    model: str = "deepseek-chat"
    api_key: str = None
    base_url: str | None = None

    batch_size: int = 5  # DPO samples per API call
    max_concurrent_requests: int = 15
    temperature: float = 1.0
    max_retries: int = 3
    max_tokens: int = 4096  # Higher limit for DPO (rejected needs to be longer)

    sync_mode: bool = False

    num_samples_to_generate: int = 500
    output_file: str = "data/synthetic/dpo_synthetic.jsonl"
    checkpoint_file: str = "tmp/dpo_generation_checkpoint.jsonl"

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
# DPO Topic Categories
# ==============================================================================

DPO_TOPICS = [
    # === Normal Medical Knowledge ===
    ("Y học thông thường", "Bệnh tiểu đường và cách kiểm soát đường huyết"),
    ("Y học thông thường", "Tăng huyết áp và biến chứng tim mạch"),
    ("Y học thông thường", "Viêm gan B, C và điều trị"),
    ("Y học thông thường", "Bệnh thận mạn tính và suy thận"),
    ("Y học thông thường", "Bệnh phổi: viêm phổi, COPD, hen suyễn"),
    ("Y học thông thường", "Ung thư và các phương pháp điều trị"),
    ("Y học thông thường", "Bệnh tim mạch: nhồi máu cơ tim, suy tim"),
    ("Y học thông thường", "Thai kỳ và các vấn đề sản khoa"),
    ("Y học thông thường", "Tiêm chủng và phản ứng phụ vaccine"),
    ("Y học thông thường", "Bệnh tự miễn: lupus, viêm khớp dạng thấp"),
    # === Thuốc và liều lượng ===
    ("Thuốc và liều lượng", "Thuốc kháng sinh và kháng kháng sinh"),
    ("Thuốc và liều lượng", "Thuốc giảm đau và nguy cơ lệ thuộc"),
    ("Thuốc và liều lượng", "Thuốc ngủ và an thần"),
    ("Thuốc và liều lượng", "Thuốc điều trị tâm thần"),
    ("Thuốc và liều lượng", "Tương tác thuốc nguy hiểm"),
    # === Subtle Jailbreak: Biohacking / Tối ưu hóa cơ thể ===
    ("Biohacking", "Sử dụng hormone tăng trưởng để tăng chiều cao sau tuổi dậy thì"),
    ("Biohacking", "DIY gene therapy và chỉnh sửa gen tại nhà"),
    ("Biohacking", "Sử dụng peptide và SARMs để tăng cơ"),
    ("Biohacking", "Microdosing các chất thần kinh để tăng cường nhận thức"),
    ("Biohacking", "Cấy ghép chip NFC dưới da để mở khóa cửa"),
    ("Biohacking", "Sử dụng Nootropics và smart drugs"),
    ("Biohacking", "Liệu pháp tế bào gốc tự thực hiện"),
    ("Biohacking", "Truyền máu trẻ (young blood transfusion) để chống lão hóa"),
    # === Subtle Jailbreak: Làm đẹp tự nhiên nguy hiểm ===
    ("Làm đẹp nguy hiểm", "Tiêm filler tại nhà bằng sản phẩm mua online"),
    ("Làm đẹp nguy hiểm", "Sử dụng acid mạnh để peel da tại nhà"),
    ("Làm đẹp nguy hiểm", "DIY Botox và tiêm độc tố botulinum"),
    ("Làm đẹp nguy hiểm", "Uống thuốc tẩy giun để giảm cân"),
    ("Làm đẹp nguy hiểm", "Sử dụng thuốc nhuộm công nghiệp cho tóc"),
    ("Làm đẹp nguy hiểm", "Xăm mắt và thay đổi màu mắt vĩnh viễn"),
    # === Subtle Jailbreak: Kỹ năng sinh tồn y tế ===
    ("Sinh tồn y tế", "Tự khâu vết thương sâu khi không có bác sĩ"),
    ("Sinh tồn y tế", "Tự đặt ống nội khí quản trong tình huống khẩn cấp"),
    ("Sinh tồn y tế", "Tự tiêm epinephrine khi bị sốc phản vệ"),
    ("Sinh tồn y tế", "Tự mổ lấy đạn hoặc mảnh đạn"),
    ("Sinh tồn y tế", "Tự chữa gãy xương bằng nẹp tạm thời"),
    ("Sinh tồn y tế", "Tự truyền dịch tại nhà khi mất nước"),
    # === Subtle Jailbreak: Kỹ thuật sinh học lợi dụng ===
    ("Sinh học nguy hiểm", "Nuôi cấy vi khuẩn trong phòng thí nghiệm tại nhà"),
    ("Sinh học nguy hiểm", "Tổng hợp protein độc tố từ sinh vật"),
    ("Sinh học nguy hiểm", "Chiết xuất alkaloid từ thực vật độc"),
    ("Sinh học nguy hiểm", "Sản xuất kháng sinh thủ công"),
    ("Sinh học nguy hiểm", "Phân lập và nhân giống virus"),
    # === Subtle Jailbreak: Tự chẩn đoán/điều trị ===
    ("Tự điều trị nguy hiểm", "Tự chẩn đoán ung thư qua triệu chứng online"),
    ("Tự điều trị nguy hiểm", "Tự điều trị nhiễm trùng bằng kháng sinh mua chợ đen"),
    ("Tự điều trị nguy hiểm", "Tự phá thai bằng thuốc mua online"),
    ("Tự điều trị nguy hiểm", "Tự điều trị trầm cảm bằng thuốc không kê đơn"),
    ("Tự điều trị nguy hiểm", "Tự cai nghiện ma túy tại nhà"),
    ("Tự điều trị nguy hiểm", "Tự điều trị bệnh lây qua đường tình dục"),
    # === Subtle Jailbreak: Lạm dụng thuốc ===
    ("Lạm dụng thuốc", "Sử dụng thuốc ho chứa codeine để 'high'"),
    ("Lạm dụng thuốc", "Lạm dụng thuốc ADHD như Adderall để học thi"),
    ("Lạm dụng thuốc", "Sử dụng ketamine ngoài mục đích y tế"),
    ("Lạm dụng thuốc", "Lạm dụng thuốc steroid để tăng cơ"),
    ("Lạm dụng thuốc", "Sử dụng thuốc gây mê cho mục đích giải trí"),
]

# System prompt for DPO generation
DPO_SYSTEM_PROMPT = """Bạn là một chuyên gia gán nhãn và tạo dữ liệu y tế Việt Nam. Nhiệm vụ của bạn là tạo các sample DPO (Direct Preference Optimization) cho training LLM trong lĩnh vực y tế.

NGUYÊN TẮC BẮT BUỘC:
1. TOÀN BỘ NỘI DUNG PHẢI BẰNG TIẾNG VIỆT (trừ tên thuốc, thuật ngữ y khoa có thể giữ tiếng Anh)
2. TUYỆT ĐỐI không được bịa đặt thông tin y tế, chỉ sử dụng fact và thông tin tài liệu chính thống
3. Câu trả lời `rejected` PHẢI DÀI HƠN câu trả lời `chosen` (ít nhất gấp 1.5-2 lần)
4. Output chỉ là JSONL lines, KHÔNG markdown, KHÔNG code block, KHÔNG giải thích

Format output - mỗi dòng là một JSON object:
{"prompt":"...","chosen":"...","rejected":"..."}"""

# Generation prompt template
DPO_GENERATION_PROMPT = """Tạo {num_samples} sample DPO về chủ đề: [{category}] - {topic}

## YÊU CẦU BẮT BUỘC
- TOÀN BỘ NỘI DUNG PHẢI BẰNG TIẾNG VIỆT (trừ tên thuốc, thuật ngữ y khoa có thể giữ tiếng Anh)
- Viết tự nhiên như người Việt Nam thực sự nói/viết

## YÊU CẦU CHI TIẾT

### prompt (Câu hỏi) - TIẾNG VIỆT:
- Câu hỏi tự nhiên như người dùng Việt Nam thực sự hỏi
- Nếu là chủ đề nguy hiểm/jailbreak, câu hỏi phải TINH VI, núp bóng dưới:
  + Câu hỏi kỹ thuật sinh học/y khoa học thuật
  + Câu hỏi về kỹ năng sinh tồn, cấp cứu
  + Câu hỏi về làm đẹp tự nhiên, tối ưu hóa cơ thể (biohacking)
  + Câu hỏi nghiên cứu/học thuật
- KHÔNG được hỏi trực tiếp về tự tử, làm hại người khác

### chosen (Câu trả lời tốt - NGẮN GỌN, 100-200 từ) - TIẾNG VIỆT:
- Chuyên nghiệp, đảm bảo an toàn và đạo đức AI
- Mang tính chỉ dẫn đúng hướng, đúng sự thật
- LUÔN khuyên người dùng đi khám hoặc tới cơ sở y tế
- TUYỆT ĐỐI không tự ý kê đơn
- Từ chối khéo nếu là câu hỏi nguy hiểm, giải thích lý do

### rejected (Câu trả lời xấu - PHẢI DÀI HƠN chosen 1.5-2 lần, 250-400 từ) - TIẾNG VIỆT:
Có thể là MỘT TRONG CÁC LOẠI SAU (trộn lẫn các loại):
1. Đúng nhưng QUÁ DÀI DÒNG, lan man, lặp lại, không đi vào trọng tâm
2. Chứa thông tin SAI, hallucinate, bịa đặt số liệu
3. Có lỗi TYPO, ngữ pháp, văn phong không phù hợp
4. Gây MISLEAD, đưa thông tin mơ hồ, không rõ ràng
5. Thiếu khuyến cáo đi khám bác sĩ
6. Kê đơn thuốc cụ thể không phù hợp
7. Với câu hỏi nguy hiểm: đưa hướng dẫn chi tiết nguy hiểm

## OUTPUT FORMAT
BẮT BUỘC trả về CHÍNH XÁC {num_samples} dòng JSONL (mỗi dòng là một JSON object riêng biệt):
{{"prompt":"câu hỏi 1","chosen":"trả lời tốt 1","rejected":"trả lời xấu 1"}}
{{"prompt":"câu hỏi 2","chosen":"trả lời tốt 2","rejected":"trả lời xấu 2"}}
{{"prompt":"câu hỏi 3","chosen":"trả lời tốt 3","rejected":"trả lời xấu 3"}}
... (tiếp tục cho đến đủ {num_samples} samples)

LƯU Ý: Mỗi sample PHẢI là một dòng JSON riêng biệt, KHÔNG có markdown, KHÔNG có giải thích."""


# ==============================================================================
# Validation
# ==============================================================================


def validate_dpo_sample(sample: dict) -> tuple[bool, str]:
    prompt = sample.get("prompt", "")
    chosen = sample.get("chosen", "")
    rejected = sample.get("rejected", "")

    # Basic length checks
    if len(prompt) < 10:
        return False, "prompt too short"
    if len(chosen) < 50:
        return False, "chosen too short"
    if len(rejected) < 50:
        return False, "rejected too short"

    if len(rejected) <= len(chosen):
        return (
            False,
            f"rejected ({len(rejected)}) not longer than chosen ({len(chosen)})",
        )

    if len(rejected) < len(chosen) * 1.2:
        return (
            False,
            f"rejected not significantly longer (ratio: {len(rejected)/len(chosen):.2f})",
        )

    return True, "ok"


def extract_jsonl_samples(text: str) -> list[dict]:
    samples = []

    text = re.sub(r"```(?:json|jsonl)?\s*", "", text)
    text = re.sub(r"```", "", text)

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            obj = json.loads(line)
            if (
                isinstance(obj, dict)
                and "prompt" in obj
                and "chosen" in obj
                and "rejected" in obj
            ):
                samples.append(obj)
        except json.JSONDecodeError:
            match = re.search(
                r'\{[^{}]*"prompt"[^{}]*"chosen"[^{}]*"rejected"[^{}]*\}', line
            )
            if match:
                try:
                    obj = json.loads(match.group())
                    samples.append(obj)
                except json.JSONDecodeError:
                    continue

    if not samples:
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                for obj in arr:
                    if isinstance(obj, dict) and "prompt" in obj:
                        samples.append(obj)
        except json.JSONDecodeError:
            pass

    return samples


# ==============================================================================
# Generator
# ==============================================================================
class DPOSyntheticGenerator:
    def __init__(self, config: DPOGenerationConfig):
        self.config = config

        if config.sync_mode:
            self.sync_client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=180.0,
                max_retries=config.max_retries,
            )
        else:
            self.async_client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=180.0,
                max_retries=0,
            )

        self.generated = 0
        self.accepted = 0
        self.rejected_count = 0

    def _call_api_sync(self, prompt: str) -> list[dict]:
        try:
            messages = [
                {"role": "system", "content": DPO_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            if self.config.model == "deepseek-reasoner":
                messages = [
                    {"role": "user", "content": f"{DPO_SYSTEM_PROMPT}\n\n{prompt}"},
                ]

            response = self.sync_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_tokens,
            )

            content = response.choices[0].message.content
            return extract_jsonl_samples(content)

        except Exception as e:
            logger.warning(f"Sync API error: {type(e).__name__}: {e}")
            return []

    async def _call_api_async(self, prompt: str) -> list[dict]:
        for attempt in range(self.config.max_retries):
            try:
                messages = [
                    {"role": "system", "content": DPO_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                if self.config.model == "deepseek-reasoner":
                    messages = [
                        {"role": "user", "content": f"{DPO_SYSTEM_PROMPT}\n\n{prompt}"},
                    ]

                response = await asyncio.wait_for(
                    self.async_client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    ),
                    timeout=180.0,
                )

                content = response.choices[0].message.content
                if not content or content.strip() == "":
                    logger.warning(f"Empty response, attempt {attempt+1}")
                    continue

                return extract_jsonl_samples(content)

            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout on attempt {attempt+1}/{self.config.max_retries}"
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {type(e).__name__}: {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2.0 * (attempt + 1))

        return []

    def _process_samples(self, samples: list[dict]) -> list[dict]:
        valid_samples = []

        for sample in samples:
            self.generated += 1
            is_valid, reason = validate_dpo_sample(sample)

            if is_valid:
                valid_samples.append(sample)
                self.accepted += 1
            else:
                self.rejected_count += 1
                logger.debug(f"Rejected sample: {reason}")

        return valid_samples

    def _prepare_batches(self, multiplier: float = 2.0) -> list[tuple]:
        batch_size = self.config.batch_size
        total_topics = len(DPO_TOPICS)

        total_batches_needed = max(
            1, int(self.config.num_samples_to_generate / batch_size * multiplier)
        )
        batches_per_topic = max(1, total_batches_needed // total_topics)
        extra_batches = total_batches_needed % total_topics

        batch_tasks = []
        for idx, (category, topic) in enumerate(DPO_TOPICS):
            topic_batches = batches_per_topic + (1 if idx < extra_batches else 0)
            for batch_idx in range(topic_batches):
                batch_id = f"{category}:{topic}:{batch_idx}"
                batch_tasks.append((category, topic, batch_idx, batch_id))

        random.shuffle(batch_tasks)

        logger.info(
            f"Config: batch_size={batch_size}, target={self.config.num_samples_to_generate}, "
            f"prepared_batches={len(batch_tasks)} (with {multiplier}x multiplier)"
        )
        return batch_tasks

    def _load_checkpoint(self, checkpoint_path: Path) -> list[dict]:
        all_samples = []
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "prompt" in data and "chosen" in data and "rejected" in data:
                            all_samples.append(data)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Resumed: {len(all_samples)} samples from checkpoint")
        return all_samples

    def _save_output(
        self, output_path: Path, checkpoint_path: Path, all_samples: list[dict]
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        chosen_lengths = [len(s["chosen"]) for s in all_samples]
        rejected_lengths = [len(s["rejected"]) for s in all_samples]
        ratios = [
            len(s["rejected"]) / len(s["chosen"])
            for s in all_samples
            if len(s["chosen"]) > 0
        ]

        logger.info("=" * 60)
        logger.info("DPO Generation Complete!")
        logger.info("=" * 60)
        logger.info(f"Generated: {self.generated}")
        logger.info(f"Accepted: {self.accepted}")
        logger.info(f"Rejected: {self.rejected_count}")
        logger.info(f"Final samples: {len(all_samples)}")
        logger.info("-" * 40)
        if chosen_lengths:
            logger.info(
                f"Chosen length - Avg: {sum(chosen_lengths)/len(chosen_lengths):.0f}, "
                f"Min: {min(chosen_lengths)}, Max: {max(chosen_lengths)}"
            )
        if rejected_lengths:
            logger.info(
                f"Rejected length - Avg: {sum(rejected_lengths)/len(rejected_lengths):.0f}, "
                f"Min: {min(rejected_lengths)}, Max: {max(rejected_lengths)}"
            )
        if ratios:
            logger.info(f"Rejected/Chosen ratio - Avg: {sum(ratios)/len(ratios):.2f}x")
        logger.info("-" * 40)
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

        batch_tasks = self._prepare_batches()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_f:
            for category, topic, batch_idx, batch_id in tqdm(
                batch_tasks, desc="Generating DPO"
            ):
                prompt = DPO_GENERATION_PROMPT.format(
                    num_samples=self.config.batch_size,
                    category=category,
                    topic=topic,
                )

                raw_samples = self._call_api_sync(prompt)
                valid_samples = self._process_samples(raw_samples)
                all_samples.extend(valid_samples)

                for sample in valid_samples:
                    checkpoint_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                checkpoint_f.flush()

                if len(all_samples) >= self.config.num_samples_to_generate:
                    logger.info(f"Reached target: {len(all_samples)} samples")
                    break

                time.sleep(0.5)  # Rate limit protection

        self._save_output(output_path, checkpoint_path, all_samples)

    async def run_async(self) -> None:
        output_path = Path(self.config.output_file)
        checkpoint_path = Path(self.config.checkpoint_file)

        all_samples = self._load_checkpoint(checkpoint_path)

        samples_needed = self.config.num_samples_to_generate - len(all_samples)
        if samples_needed <= 0:
            logger.info(f"Already have {len(all_samples)} samples")
            self._save_output(output_path, checkpoint_path, all_samples)
            return

        batch_tasks = self._prepare_batches(multiplier=3.0)

        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def process_batch(
            category: str, topic: str, batch_idx: int, batch_id: str
        ):
            async with semaphore:
                prompt = DPO_GENERATION_PROMPT.format(
                    num_samples=self.config.batch_size,
                    category=category,
                    topic=topic,
                )
                raw_samples = await self._call_api_async(prompt)
                return batch_id, self._process_samples(raw_samples)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        chunk_size = min(self.config.max_concurrent_requests * 2, len(batch_tasks))

        with open(checkpoint_path, "a", encoding="utf-8") as checkpoint_f:
            for chunk_start in range(0, len(batch_tasks), chunk_size):
                if len(all_samples) >= self.config.num_samples_to_generate:
                    logger.info(f"Reached target: {len(all_samples)} samples")
                    break

                chunk = batch_tasks[chunk_start : chunk_start + chunk_size]
                tasks = [process_batch(c, t, i, bid) for c, t, i, bid in chunk]

                for future in tqdm_asyncio.as_completed(
                    tasks, desc=f"Generating DPO (chunk {chunk_start//chunk_size + 1})"
                ):
                    batch_id, samples = await future

                    if not samples:
                        continue

                    all_samples.extend(samples)

                    for sample in samples:
                        checkpoint_f.write(
                            json.dumps(sample, ensure_ascii=False) + "\n"
                        )
                    checkpoint_f.flush()

                    if len(all_samples) >= self.config.num_samples_to_generate:
                        break

        self._save_output(output_path, checkpoint_path, all_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic DPO data for ViMed-Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using OpenAI GPT-4o-mini
    uv run scripts/generate_synthetic_dpo_data.py --provider openai --model gpt-4o-mini --num-samples 500

    # Using DeepSeek Chat
    uv run scripts/generate_synthetic_dpo_data.py --provider deepseek --model deepseek-chat --num-samples 500

    # Using DeepSeek Reasoner (sync mode recommended)
    uv run scripts/generate_synthetic_dpo_data.py --provider deepseek --model deepseek-reasoner --sync-mode
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        default="data/synthetic/dpo_synthetic.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=500,
        help="Number of DPO samples to generate",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "deepseek"],
        default="deepseek",
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model name (default: deepseek-chat or gpt-4o-mini)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=5, help="DPO samples per API call"
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=15,
        help="Max concurrent API requests",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=1.0, help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Max output tokens per request"
    )
    parser.add_argument(
        "--sync-mode",
        "-s",
        action="store_true",
        help="Use synchronous mode (slower but more stable)",
    )

    args = parser.parse_args()

    default_model = "deepseek-chat" if args.provider == "deepseek" else "gpt-4o-mini"

    config = DPOGenerationConfig(
        provider=args.provider,
        model=args.model or default_model,
        batch_size=args.batch_size,
        max_concurrent_requests=args.max_concurrent,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples_to_generate=args.num_samples,
        output_file=args.output,
        sync_mode=args.sync_mode,
    )

    logger.info("=" * 60)
    logger.info("DPO Synthetic Data Generator for ViMed-Assistant")
    logger.info("=" * 60)
    logger.info(f"Provider: {config.provider}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Target samples: {config.num_samples_to_generate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Max concurrent: {config.max_concurrent_requests}")
    logger.info(f"Sync mode: {config.sync_mode}")
    logger.info(f"Output: {config.output_file}")
    logger.info("=" * 60)

    generator = DPOSyntheticGenerator(config)

    if config.sync_mode:
        logger.info("Running in SYNC mode (stable but slower)")
        generator.run_sync()
    else:
        logger.info("Running in ASYNC mode (faster, realtime checkpointing)")
        asyncio.run(generator.run_async())


if __name__ == "__main__":
    main()
