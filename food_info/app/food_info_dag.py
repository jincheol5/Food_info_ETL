from airflow import DAG
from food_info.module import DBInterface

dag=DAG(
    dag_id="nutrition_extractor",
    schedule=None
)

