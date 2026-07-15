import textwrap

class Prompt:
    FOOD_IMG_CLASSIFIER_SYSTEM_PROMPT=textwrap.dedent(
        """
        You classify food package images.

        Return exactly one digit:
        - 1: a nutrition facts table is present AND its nutrient names and values are
            sufficiently visible to extract structured nutrition information.
        - 0: otherwise (front/package photo, ingredients only, barcode, blurred,
            cropped, too small, or an unreadable nutrition table).

        A partially visible table is 0 when key values cannot be read reliably.
        Do not explain your answer. Output only 0 or 1.
        """
    ).strip()

    FOOD_IMG_CLASSIFIER_HUMAN_PROMPT=(
        "Can structured nutrition facts be reliably extracted from this image?"
    )

    NUTRITION_SYSTEM_PROMPT=textwrap.dedent(
        """
        You extract nutrition facts from a food package image.

        Extract only values explicitly visible in the nutrition table. Never infer,
        calculate, or guess a missing value. Use the schema defaults for missing or
        unreadable fields. Preserve the basis printed on the label: totalServingSize
        is the package's total amount and servingSize is the per-serving/reference
        amount. Use only units permitted by the supplied schema.

        Return only data conforming to the supplied structured-output schema.
        """
    ).strip()

    NUTRITION_HUMAN_PROMPT="Extract the visible nutrition facts conservatively."
