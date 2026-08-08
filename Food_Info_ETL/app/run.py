import argparse
from module import FoodInfoChain,DBInterface
from schema import NutritionSchema
from utils import DataUtils

def run_app(**kwargs):
    """
    Process DB documents that have not yet received nutrition data.
    """
    classifier=FoodInfoChain.get_food_img_classify_chain(kwargs["model_name"],kwargs["ollama_port"])
    extractor=FoodInfoChain.get_nutrition_extract_chain(kwargs["model_name"],kwargs["ollama_port"])
    db=DBInterface(port=kwargs["db_port"])

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

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="gemma4:latest")
    parser.add_argument("--ollama_port",type=int,default=11434)
    parser.add_argument("--db_port",type=int,default=27017)
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "ollama_port":args.ollama_port,
        "db_port":args.db_port
    }
    run_app(**app_config)
