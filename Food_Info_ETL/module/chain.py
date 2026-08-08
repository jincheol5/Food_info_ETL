from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import ValidationError
from .model_utils import ModelUtils
from schema import NutritionSchema


class FoodInfoChain:
    @staticmethod
    def _model(model_name:str,port:int)->ChatOllama:
        return ChatOllama(
            model=model_name,
            base_url=f"http://127.0.0.1:{port}",
            temperature=0,
        )

    @classmethod
    def get_food_img_classify_chain(
            cls, 
            model_name:str="gemma4", 
            port:int=11434
        ):
        chain=(
            RunnableLambda(ModelUtils.get_classifier_message)
            | cls._model(model_name,port)
            | RunnableLambda(ModelUtils.parse_classifier_output)
        )
        return chain.with_retry(
            stop_after_attempt=3, 
            retry_if_exception_type=(ValueError,)
        )

    @classmethod
    def get_nutrition_extract_chain(
            cls, 
            model_name:str="gemma4", 
            port:int=11434
        ):
        model=cls._model(model_name,port).with_structured_output(NutritionSchema)
        chain=RunnableLambda(ModelUtils.get_nutrition_message) | model
        return chain.with_retry(
            stop_after_attempt=3,
            retry_if_exception_type=(ValidationError, ValueError),
        )
