# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

import json
import random
import os

# Load geometry metadata for curriculum training
def load_geometry_metadata(metadata_path: str = None) -> list:
    """Load geometry metadata from JSON file."""
    if metadata_path is None:
        # Default path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        metadata_path = os.path.join(base_dir, "utils", "metadata_all.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            data = json.load(f)
            return data.get("metadata", [])
    return []


def format_geometry_for_prompt(geom_meta: dict) -> str:
    """Format geometry metadata into a prompt string for the curriculum agent."""
    filepath = geom_meta["file"]
    filename = os.path.basename(filepath)

    # Summarize surface types
    entity_types = geom_meta.get("entity_types", {})
    type_counts = {}
    for surf_id, surf_type in entity_types.items():
        type_counts[surf_type] = type_counts.get(surf_type, 0) + 1
    type_summary = ", ".join([f"{count} {t}" for t, count in sorted(type_counts.items())])

    return (
        f"**GEOMETRY FILE:** {filename}\n"
        f"**GEOMETRY PATH:** {filepath}\n"
        f"**TOPOLOGY:** {geom_meta['surfaces']} surfaces, {geom_meta['volumes']} volume(s), "
        f"{geom_meta['points']} points, {geom_meta['curves']} curves\n"
        f"**SURFACE TYPES:** {type_summary}\n\n"
        f"To load this geometry in Gmsh:\n"
        f"```python\n"
        f"gmsh.model.occ.importShapes(\"{filepath}\")\n"
        f"gmsh.model.occ.synchronize()\n"
        f"```"
    )


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}



def process_image(image: Union[Dict[str, Any], ImageObject, str], min_pixels: int, max_pixels: int) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if "questioner_format_with_persona" in self.format_prompt:
            print("load personas")
            personas_dataset = load_dataset("proj-persona/PersonaHub", "math", split="train")
            self.personas = [item['input persona'] for item in personas_dataset]
            # self.personas = self.personas.select(range(100))

        # Load geometry metadata for GMSH curriculum training
        self.geometry_metadata = []
        if self.format_prompt and "gmsh_format" in self.format_prompt:
            self.geometry_metadata = load_geometry_metadata()
            print(f"Loaded {len(self.geometry_metadata)} geometries for GMSH curriculum training")

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        # GMSH curriculum: agent creates meshing task for a PROVIDED geometry
        if "gmsh_format" in self.format_prompt:
            # Select a random geometry from loaded metadata
            if self.geometry_metadata:
                geom_meta = random.choice(self.geometry_metadata)
                geometry_context = format_geometry_for_prompt(geom_meta)
            else:
                geometry_context = "No geometry available. Create your own simple geometry."

            return [
                {
                    "role": "system",
                    "content": (
                        "You are a curriculum designer for Gmsh meshing training.\n\n"
                        "You will be given a STEP geometry file. Your job is to create a challenging meshing task for this geometry.\n\n"
                        "Your task MUST specify:\n"
                        "1. The analysis type (thermal, structural, modal, CFD, acoustic, electromagnetic, etc.)\n"
                        "2. Physical groups with meaningful names based on the geometry's surfaces\n"
                        "   - Identify surfaces by their type (Plane, Cylinder, Torus, etc.) and assign appropriate boundary conditions\n"
                        "   - Examples: 'inlet', 'outlet', 'wall', 'fixed_support', 'heat_source', 'symmetry_plane'\n"
                        "3. Mesh size requirements:\n"
                        "   - Global mesh size\n"
                        "   - Local refinement near curved surfaces, small features, or boundary layers\n"
                        "   - Use mesh size fields (Distance, Threshold, Box) for advanced refinement\n"
                        "4. Mesh quality requirements if applicable (element order, optimization)\n\n"
                        "DIFFICULTY LEVELS to vary:\n"
                        "- BASIC: Simple mesh with uniform size, 2-3 physical groups\n"
                        "- INTERMEDIATE: Multiple physical groups, local refinement near one region\n"
                        "- ADVANCED: Boundary layers, multiple refinement zones, transfinite meshing\n"
                        "- EXPERT: Anisotropic mesh, curvature-based sizing, structured regions\n\n"
                        "Output exactly:\n"
                        "<task>[complete meshing task description including the geometry path]</task>\n\n"
                        "Example:\n"
                        "<task>Load the geometry from [path]. Mesh it for thermal analysis. Create physical groups: 'heat_source' for the cylindrical surfaces (apply heat flux), 'convection_surfaces' for planar faces (convective cooling), 'volume' for the solid domain. Use global element size 2.0 with refinement to 0.5 near cylindrical surfaces using a Distance field.</task>"
                    )
                },
                {
                    "role": "user",
                    "content": f"Create a meshing task for this geometry:\n\n{geometry_context}"
                }
            ]
        prompt_str: str = example[self.prompt_key]
        if "questioner_format_with_persona" in self.format_prompt:
            print("load personas")
            return [
                {
                    "role": "system",
                    "content": (
                        f"You are {random.choice(self.personas)}.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                        "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                        "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging reasoning question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        if "questioner_format" in self.format_prompt:
            # print('detected questioner_format')
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an expert competition-math problem setter.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                        "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                        "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging reasoning question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        if "solver_format" in self.format_prompt:
            return [
                {
                    "role": "system", 
                    "content": r"Please reason step by step, and put your final answer within \boxed{}."
                },
                {
                    "role": "user", 
                    "content": prompt_str
                }
                ]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # Pass all fields from example (including metadata) to Jinja template
            prompt_str = format_prompt.render(content=prompt_str, **example)
        
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        if self.tokenizer.chat_template:
            return (
                len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
            )
        else:
            return (
                len("system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]) <= self.max_prompt_length
            )
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            raw_image_data = example.pop(self.image_key)
            images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                for image in raw_image_data
            ]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": raw_image_data}
        else:
            if self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = "system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
