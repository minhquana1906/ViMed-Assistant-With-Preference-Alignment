"""
Data Correction Script for ViMed-Assistant SFT Dataset.

This script uses LLM APIs (OpenAI gpt-4o-mini or DeepSeek-chat) to:
1. Fix spelling errors in Vietnamese medical content
2. Reformat data for consistency
3. Apply guardrails to ensure accuracy and safety

Key Features:
- Batch processing for cost & time efficiency
- Multiple guardrails to prevent hallucination
- Semantic similarity check to ensure minimal content change
- Medical term preservation
- Checkpoint/resume support

Usage:
    uv run scripts/correct_data.py \
    --input data/sft.jsonl \
    --output data/sft_corrected.jsonl \
    --provider openai \
    --model gpt-4o-mini \
    --batch-size 20
"""

import argparse
import asyncio
import json
import logging
import os
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_correction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class CorrectionConfig:
    """Configuration for data correction pipeline."""

    # API Settings
    provider: Literal["openai", "deepseek"] = "deepseek"
    model: str = "deepseek-chat"  # or "gpt-4o-mini"
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str | None = None  # For DeepSeek: "https://api.deepseek.com"

    # Processing Settings
    batch_size: int = 10  # Number of samples per batch request
    max_concurrent_requests: int = 5  # Concurrent API requests
    temperature: float = 0.1  # Low temperature for deterministic output
    max_retries: int = 3
    retry_delay: float = 1.0

    # Guardrail Thresholds
    max_content_change_ratio: float = 0.3  # Max 30% content change allowed
    min_similarity_score: float = 0.85  # Minimum similarity required
    max_length_change_ratio: float = 0.2  # Max 20% length change

    # Paths
    input_file: str = "data/sft.jsonl"
    output_file: str = "data/sft_corrected.jsonl"
    checkpoint_file: str = "tmp/correction_checkpoint.json"
    rejected_file: str = "tmp/correction_rejected.jsonl"

    def __post_init__(self):
        """Set provider-specific defaults."""
        if self.provider == "deepseek":
            self.base_url = self.base_url or "https://api.deepseek.com"
            self.model = self.model or "deepseek-chat"
            if not self.api_key:
                self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        elif self.provider == "openai":
            self.model = self.model or "gpt-4o-mini"
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY", "")


# ==============================================================================
# Guardrails
# ==============================================================================
class CorrectionGuardrails:
    """Guardrails to ensure safe and accurate data correction."""

    # Vietnamese medical terms that MUST be preserved (not modified)
    PROTECTED_MEDICAL_TERMS = {
        # Disease names
        "ung thư",
        "tiểu đường",
        "huyết áp",
        "tim mạch",
        "viêm gan",
        "HIV",
        "AIDS",
        "COVID-19",
        "SARS-CoV-2",
        "lao phổi",
        "viêm phổi",
        "viêm họng",
        "viêm amidan",
        "trào ngược",
        "sỏi thận",
        "sỏi niệu quản",
        "tiền đình",
        "đột quỵ",
        # Medical procedures
        "sinh thiết",
        "nội soi",
        "siêu âm",
        "X-quang",
        "CT",
        "MRI",
        "xét nghiệm máu",
        "tán sỏi",
        "phẫu thuật",
        "mổ",
        # Drug names
        "insulin",
        "paracetamol",
        "ibuprofen",
        "aspirin",
        # Lab values
        "G/L",
        "mg/dL",
        "mmHg",
        "mmol/L",
        "%",
        # Anatomical terms
        "amidan",
        "niệu quản",
        "thận",
        "gan",
        "phổi",
        "tim",
    }

    # Patterns that indicate medical measurements (should not be modified)
    MEASUREMENT_PATTERNS = [
        r"\d+(?:\.\d+)?\s*(?:G/L|mg/dL|mmHg|mmol/L|%|mm|cm|kg|g)",
        r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?",  # Ranges like "0.9-2.9"
        r"\d+(?:,\d+)?\s*độ\s*C(?:elsius)?",  # Temperature
    ]

    @classmethod
    def extract_protected_content(cls, text: str) -> set[str]:
        """Extract protected terms and patterns from text."""
        protected = set()

        # Find protected medical terms
        text_lower = text.lower()
        for term in cls.PROTECTED_MEDICAL_TERMS:
            if term.lower() in text_lower:
                protected.add(term.lower())

        # Find measurement patterns
        for pattern in cls.MEASUREMENT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            protected.update(m.lower() for m in matches)

        return protected

    @classmethod
    def validate_protected_content(
        cls, original: str, corrected: str
    ) -> tuple[bool, str]:
        """Validate that protected content is preserved."""
        original_protected = cls.extract_protected_content(original)
        corrected_protected = cls.extract_protected_content(corrected)

        missing = original_protected - corrected_protected
        if missing:
            return False, f"Missing protected terms: {missing}"

        return True, "OK"

    @classmethod
    def calculate_similarity(cls, original: str, corrected: str) -> float:
        """Calculate character-level similarity between texts."""
        if not original or not corrected:
            return 0.0

        # Simple character-level similarity
        original_chars = set(original.lower())
        corrected_chars = set(corrected.lower())

        intersection = len(original_chars & corrected_chars)
        union = len(original_chars | corrected_chars)

        return intersection / union if union > 0 else 0.0

    @classmethod
    def calculate_change_ratio(cls, original: str, corrected: str) -> float:
        """Calculate the ratio of changes made."""
        if not original:
            return 1.0

        # Count character-level differences
        original_words = set(original.lower().split())
        corrected_words = set(corrected.lower().split())

        added = corrected_words - original_words
        removed = original_words - corrected_words

        total_changes = len(added) + len(removed)
        total_original = len(original_words)

        return total_changes / total_original if total_original > 0 else 0.0

    @classmethod
    def validate_length_change(
        cls, original: str, corrected: str, max_ratio: float = 0.2
    ) -> tuple[bool, str]:
        """Validate length change is within acceptable range."""
        if not original:
            return True, "OK"

        original_len = len(original)
        corrected_len = len(corrected)

        change_ratio = abs(corrected_len - original_len) / original_len

        if change_ratio > max_ratio:
            return (
                False,
                f"Length change {change_ratio:.2%} exceeds max {max_ratio:.2%}",
            )

        return True, "OK"

    @classmethod
    def validate_no_new_medical_claims(
        cls, original: str, corrected: str
    ) -> tuple[bool, str]:
        """Ensure no new medical claims or diagnoses are added."""
        # Keywords that indicate medical claims
        claim_keywords = [
            "chẩn đoán",
            "bệnh",
            "điều trị",
            "thuốc",
            "liều",
            "uống",
            "tiêm",
            "phẫu thuật",
            "triệu chứng",
            "nguyên nhân",
        ]

        original_lower = original.lower()
        corrected_lower = corrected.lower()

        for keyword in claim_keywords:
            # Count occurrences
            original_count = original_lower.count(keyword)
            corrected_count = corrected_lower.count(keyword)

            # If new claims are added
            if corrected_count > original_count + 1:  # Allow 1 word of tolerance
                return False, f"Potential new medical claim added (keyword: {keyword})"

        return True, "OK"


# ==============================================================================
# Correction Prompts
# ==============================================================================
SYSTEM_PROMPT = """Bạn là một chuyên gia ngôn ngữ học y tế, với nhiệm vụ sửa lỗi chính tả và format văn bản y tế tiếng Việt.

## NHIỆM VỤ
Sửa lỗi chính tả, lỗi đánh máy, và format lại văn bản cho đẹp hơn.

## QUY TẮC BẮT BUỘC
1. KHÔNG thay đổi ý nghĩa nội dung
2. KHÔNG thêm thông tin mới
3. KHÔNG xóa thông tin quan trọng
4. GIỮ NGUYÊN tất cả thuật ngữ y tế, tên bệnh, tên thuốc, chỉ số xét nghiệm
5. GIỮ NGUYÊN các con số, đơn vị đo lường
6. CHỈ sửa lỗi chính tả, dấu câu, và format
7. Nếu có ký tự tiếng Trung, xóa hoặc thay thế bằng tiếng Việt phù hợp
8. Đảm bảo format phù hợp cho markdown, đặc biệt tại các headings, lists, code blocks, bulleted points

## VÍ DỤ SỬA LỖI
- "bệnh tiẻu đường" → "bệnh tiểu đường"
- "viem gan b" → "viêm gan B"
- "bị đau chân" → "Tôi bị đau chân"
- "- Khó tho- Tuc ngực" → "\n- Khó thở\n- Tức ngực"

## OUTPUT FORMAT
Trả về JSON với format:
{
    "corrected_content": "Nội dung đã sửa",
    "changes": ["Liệt kê các thay đổi đã thực hiện"],
    "confidence": 0.95
}

Nếu không cần sửa gì, trả về nội dung gốc với changes=[] và confidence=1.0"""

BATCH_PROMPT_TEMPLATE = """Sửa lỗi chính tả và format cho các đoạn văn y tế sau.

Với MỖI đoạn văn, trả về JSON object với:
- "id": ID của đoạn văn
- "corrected_content": Nội dung đã sửa
- "changes": Danh sách các thay đổi
- "confidence": Độ tin cậy (0-1)

## CÁC ĐOẠN VĂN CẦN SỬA

{batch_content}

## OUTPUT
Trả về một JSON array chứa kết quả cho từng đoạn văn, theo thứ tự."""


# ==============================================================================
# Data Correction Pipeline
# ==============================================================================
class DataCorrectionPipeline:
    """Pipeline for correcting SFT data with guardrails."""

    def __init__(self, config: CorrectionConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.guardrails = CorrectionGuardrails()
        self.stats = {
            "total": 0,
            "corrected": 0,
            "unchanged": 0,
            "rejected": 0,
            "errors": 0,
        }

    async def correct_batch(self, batch: list[dict], batch_id: int) -> list[dict]:
        """Correct a batch of samples using LLM API."""
        # Prepare batch content for prompt
        batch_content = ""
        for i, sample in enumerate(batch):
            # Extract assistant content for correction
            messages = sample.get("messages", [])
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    batch_content += f"### ID: {i}\n{content}\n\n"

        prompt = BATCH_PROMPT_TEMPLATE.format(batch_content=batch_content)

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"},
                )

                result_text = response.choices[0].message.content
                results = json.loads(result_text)

                # Handle both array and object responses
                if isinstance(results, dict):
                    if "results" in results:
                        results = results["results"]
                    elif "corrected_content" in results:
                        results = [results]

                return self._process_batch_results(batch, results)

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Batch {batch_id} attempt {attempt+1}: JSON decode error: {e}"
                )
            except Exception as e:
                logger.warning(f"Batch {batch_id} attempt {attempt+1}: {e}")
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        # Return original samples if all retries failed
        logger.error(f"Batch {batch_id}: All retries failed, keeping original")
        return batch

    def _process_batch_results(
        self, batch: list[dict], results: list[dict]
    ) -> list[dict]:
        """Process and validate batch results with guardrails."""
        processed = []

        for i, sample in enumerate(batch):
            try:
                # Find corresponding result
                result = next(
                    (r for r in results if r.get("id") == i),
                    results[i] if i < len(results) else None,
                )

                if not result:
                    processed.append(sample)
                    self.stats["unchanged"] += 1
                    continue

                corrected_content = result.get("corrected_content", "")
                confidence = result.get("confidence", 0)

                # Get original assistant content
                original_content = ""
                messages = sample.get("messages", [])
                for msg in messages:
                    if msg.get("role") == "assistant":
                        original_content = msg.get("content", "")
                        break

                # Apply guardrails
                validation_result = self._validate_correction(
                    original_content, corrected_content, confidence
                )

                if validation_result["valid"]:
                    # Apply correction
                    new_sample = self._apply_correction(sample, corrected_content)
                    processed.append(new_sample)
                    if result.get("changes"):
                        self.stats["corrected"] += 1
                    else:
                        self.stats["unchanged"] += 1
                else:
                    # Keep original, log rejection
                    processed.append(sample)
                    self.stats["rejected"] += 1
                    logger.warning(
                        f"Rejected correction: {validation_result['reason']}"
                    )

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                processed.append(sample)
                self.stats["errors"] += 1

        return processed

    def _validate_correction(
        self, original: str, corrected: str, confidence: float
    ) -> dict:
        """Validate correction against all guardrails."""
        if not corrected:
            return {"valid": False, "reason": "Empty corrected content"}

        # Guardrail 1: Confidence threshold
        if confidence < 0.8:
            return {"valid": False, "reason": f"Low confidence: {confidence}"}

        # Guardrail 2: Protected content preservation
        valid, msg = self.guardrails.validate_protected_content(original, corrected)
        if not valid:
            return {"valid": False, "reason": msg}

        # Guardrail 3: Length change
        valid, msg = self.guardrails.validate_length_change(
            original, corrected, self.config.max_length_change_ratio
        )
        if not valid:
            return {"valid": False, "reason": msg}

        # Guardrail 4: Content change ratio
        change_ratio = self.guardrails.calculate_change_ratio(original, corrected)
        if change_ratio > self.config.max_content_change_ratio:
            return {
                "valid": False,
                "reason": f"Content change {change_ratio:.2%} exceeds limit",
            }

        # Guardrail 5: Similarity check
        similarity = self.guardrails.calculate_similarity(original, corrected)
        if similarity < self.config.min_similarity_score:
            return {
                "valid": False,
                "reason": f"Low similarity: {similarity:.2%}",
            }

        # Guardrail 6: No new medical claims
        valid, msg = self.guardrails.validate_no_new_medical_claims(original, corrected)
        if not valid:
            return {"valid": False, "reason": msg}

        return {"valid": True, "reason": "Passed all guardrails"}

    def _apply_correction(self, sample: dict, corrected_content: str) -> dict:
        """Apply correction to sample."""
        new_sample = json.loads(json.dumps(sample))  # Deep copy
        for msg in new_sample.get("messages", []):
            if msg.get("role") == "assistant":
                msg["content"] = corrected_content
                break
        return new_sample

    async def process_file(self) -> None:
        """Process entire file with batch processing."""
        input_path = Path(self.config.input_file)
        output_path = Path(self.config.output_file)
        checkpoint_path = Path(self.config.checkpoint_file)

        # Load data
        logger.info(f"Loading data from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        self.stats["total"] = len(data)
        logger.info(f"Loaded {len(data)} samples")

        # Load checkpoint if exists
        start_idx = 0
        processed_data = []
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            start_idx = checkpoint.get("last_processed_idx", 0)
            processed_data = checkpoint.get("processed_data", [])
            logger.info(f"Resuming from checkpoint at index {start_idx}")

        # Create batches
        batches = []
        for i in range(start_idx, len(data), self.config.batch_size):
            batch = data[i : i + self.config.batch_size]
            batches.append((i, batch))

        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def process_with_semaphore(batch_id: int, batch: list[dict]):
            async with semaphore:
                return await self.correct_batch(batch, batch_id)

        # Process all batches
        logger.info(f"Processing {len(batches)} batches...")
        results = []

        for batch_id, batch in tqdm_asyncio(batches, desc="Correcting data"):
            result = await process_with_semaphore(batch_id, batch)
            results.extend(result)

            # Save checkpoint periodically
            if batch_id % 10 == 0:
                checkpoint_data = {
                    "last_processed_idx": batch_id + len(batch),
                    "processed_data": processed_data + results,
                    "timestamp": datetime.now().isoformat(),
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False)

        # Combine with previously processed data
        all_processed = processed_data + results

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in all_processed:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Remove checkpoint after successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # Log statistics
        logger.info("=" * 50)
        logger.info("Correction Statistics:")
        logger.info(f"  Total samples: {self.stats['total']}")
        logger.info(f"  Corrected: {self.stats['corrected']}")
        logger.info(f"  Unchanged: {self.stats['unchanged']}")
        logger.info(f"  Rejected: {self.stats['rejected']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"Output saved to: {output_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Correct spelling and format in SFT dataset"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/final_sft_cleaned.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/final_sft_corrected.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "deepseek"],
        default="deepseek",
        help="API provider (default: deepseek)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model name (default: depends on provider)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="Batch size for API requests (default: 10)",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=10,
        help="Max concurrent API requests (default: 10)",
    )

    args = parser.parse_args()

    config = CorrectionConfig(
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        max_concurrent_requests=args.max_concurrent,
        input_file=args.input,
        output_file=args.output,
    )

    pipeline = DataCorrectionPipeline(config)
    asyncio.run(pipeline.process_file())


if __name__ == "__main__":
    main()
