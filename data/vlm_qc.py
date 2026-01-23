import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


class VLMJudge:
    """Layer 4: Semantic QC using VLM-as-Judge"""

    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf", device="cuda"):
        print(f"Loading VLM Judge: {model_name}")

        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        self.device = device

    def validate_label(self, image, objects, predicted_label):
        """
        Ask VLM: "Given this scene, is the label '{predicted_label}' correct?"
        Returns: (is_valid: bool, confidence: float, reason: str)
        """
        # Build context prompt
        obj_list = ", ".join([o["category"] for o in objects])

        # Simple prompt without chat template
        prompt = f"""[INST] <image>
You are validating autonomous driving labels.
Objects detected: {obj_list}
Question: Can the vehicle move forward?
Proposed Label: {predicted_label}

Is this label correct? Answer ONLY with:
- "CORRECT" if label matches the scene
- "WRONG: [reason]" if label is incorrect [/INST]"""

        # Process inputs - simplified
        try:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",  # type: ignore
            )
        except TypeError:
            # Fallback for older API
            inputs = self.processor(
                prompt,
                image,
                return_tensors="pt",  # type: ignore
            )

        # Move to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,  # type: ignore
                max_new_tokens=50,
                do_sample=False,
            )

        # Decode - handle both old and new API
        try:
            # New API
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][input_len:]
            response = self.processor.decode(generated_ids, skip_special_tokens=True)
        except Exception:  # noqa: E722
            # Old API
            response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ]

        # Parse response
        is_valid = "CORRECT" in response.upper()
        confidence = 0.9 if is_valid else 0.3
        reason = response.split("WRONG:")[-1].strip() if not is_valid else "Valid"

        return is_valid, confidence, reason
