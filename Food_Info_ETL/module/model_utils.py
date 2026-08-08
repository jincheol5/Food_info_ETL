import base64
import mimetypes
from pathlib import Path
from langchain_core.messages import HumanMessage,SystemMessage
from schema import Prompt


class ModelUtils:
    @staticmethod
    def image_to_data_url(image_path:str)->str:
        path=Path(image_path)
        mime_type=mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded=base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @classmethod
    def _message(
            cls, 
            image_path:str, 
            system_prompt:str, 
            human_prompt:str
        ):
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": human_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": cls.image_to_data_url(image_path)},
                    },
                ]
            ),
        ]

    @classmethod
    def get_classifier_message(
            cls, 
            image_path:str
        ):
        return cls._message(
            image_path,
            Prompt.FOOD_IMG_CLASSIFIER_SYSTEM_PROMPT,
            Prompt.FOOD_IMG_CLASSIFIER_HUMAN_PROMPT,
        )

    @classmethod
    def get_nutrition_message(
            cls, 
            image_path:str
        ):
        return cls._message(
            image_path,
            Prompt.NUTRITION_SYSTEM_PROMPT,
            Prompt.NUTRITION_HUMAN_PROMPT,
        )

    @staticmethod
    def parse_classifier_output(message)->bool:
        content=message.content if hasattr(message,"content") else str(message)
        value=str(content).strip()
        if value not in {"0","1"}:
            raise ValueError(f"Classifier must return 0 or 1, got {value!r}")
        return value=="1"
