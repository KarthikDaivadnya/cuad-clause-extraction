"""
few_shot_examples.py
---------------------
Hand-written few-shot examples for each clause type. These are shown to the
LLM before the real contract passage so it learns the expected level of
detail and the exact output format. This directly addresses the assignment's
bonus: "Experiment with few-shot examples to improve clause extraction."

Each example is a (contract_passage, expected_json_answer) pair. Keep these
short and clearly representative — too many/too long examples burn context
budget for no accuracy gain.
"""

FEW_SHOT_EXAMPLES = {
    "termination_clause": [
        {
            "passage": (
                "Either party may terminate this Agreement upon thirty (30) days' "
                "prior written notice to the other party. Company may also "
                "terminate immediately for cause if Vendor breaches any material "
                "term of this Agreement and fails to cure such breach within "
                "fifteen (15) days of receiving written notice."
            ),
            "answer": {
                "found": True,
                "clause_text": "Either party may terminate this Agreement upon thirty (30) days' prior written notice; Company may terminate immediately for cause if Vendor fails to cure a material breach within fifteen (15) days of notice.",
                "notice_period": "30 days (60/15 days for cause with cure period)",
                "termination_for_cause": True,
            },
        }
    ],
    "confidentiality_clause": [
        {
            "passage": (
                "Each party agrees to hold the other party's Confidential "
                "Information in strict confidence and not to disclose it to any "
                "third party without prior written consent, except as required "
                "by law. This obligation survives termination of this Agreement "
                "for a period of five (5) years."
            ),
            "answer": {
                "found": True,
                "clause_text": "Each party must hold the other's Confidential Information in strict confidence and may not disclose it to third parties without prior written consent, except as required by law; the obligation survives termination for five (5) years.",
                "survives_termination": True,
                "duration": "5 years",
            },
        }
    ],
    "liability_clause": [
        {
            "passage": (
                "IN NO EVENT SHALL EITHER PARTY'S TOTAL LIABILITY ARISING OUT OF "
                "THIS AGREEMENT EXCEED THE TOTAL FEES PAID BY CLIENT IN THE "
                "TWELVE (12) MONTHS PRECEDING THE CLAIM. NEITHER PARTY SHALL BE "
                "LIABLE FOR ANY INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES."
            ),
            "answer": {
                "found": True,
                "clause_text": "Total liability of either party is capped at fees paid in the preceding twelve (12) months; neither party is liable for indirect, incidental, or consequential damages.",
                "liability_cap": "12 months of fees paid",
                "excludes_consequential_damages": True,
            },
        }
    ],
}
