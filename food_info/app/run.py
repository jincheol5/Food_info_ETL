import argparse
from pathlib import Path
from typing import Optional
from module import FoodInfoChain,DBInterface
from schema import NutritionSchema
from utils import DataUtils


def run_workflow(
        model_name:str="gemma4",
        ollama_port:int=11434,
        db_port:int=27017
    ):
    """
    Process DB documents that have not yet received nutrition data.
    """

    classifier=FoodInfoChain.get_food_img_classify_chain(model_name, ollama_port)
    extractor=FoodInfoChain.get_nutrition_extract_chain(model_name, ollama_port)
    db=DBInterface(port=db_port)

    try:
        food_ids=db.get_unextracted_food()

        for food_id in food_ids:
            image_paths=DataUtils.get_food_image_paths(food_id)
            nutrition_image=None
            for image_path in image_paths:
                if classifier.invoke(image_path):
                    nutrition_image=image_path
                    break

            if nutrition_image is None:
                continue

            extracted=extractor.invoke(nutrition_image)
            nutrition_info=NutritionSchema.model_validate(extracted)
            db.update_nutrition_info(
                [(food_id,nutrition_info.model_dump(mode="json"))]
            )
    finally:
        db.disconnect_db()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gemma4")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--db-port", type=int, default=27017)
    parser.add_argument("--dataset-path")
    args=parser.parse_args()
    run_workflow(args.model_name, args.ollama_port, args.db_port, args.dataset_path)


if __name__ == "__main__":
    main()
