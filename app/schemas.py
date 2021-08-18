"""
https://pydantic-docs.helpmanual.io/usage/schema/
"""


from pydantic import BaseModel, Field


feature_names = [
    "credit_card_transactions__V1",
    "credit_card_transactions__V2",
    "credit_card_transactions__V3",
    "credit_card_transactions__V4",
    "credit_card_transactions__V5",
    "credit_card_transactions__Time",
    "credit_card_transactions__Amount",
]


class CreditTransactions(BaseModel):
    credit_card_transactions__V1: float = Field(
        ..., ge=-55.40751, description="PCA Dimensionality reduction column"
    )
    credit_card_transactions__V2: float = Field(
        ..., ge=-71.71573, description="PCA Dimensionality reduction column"
    )
    credit_card_transactions__V3: float = Field(
        ..., ge=-47.32559, description="PCA Dimensionality reduction column"
    )
    credit_card_transactions__V4: float = Field(
        ..., ge=-4.683171, description="PCA Dimensionality reduction column"
    )
    credit_card_transactions__V5: float = Field(
        ..., ge=-112.7433, description="PCA Dimensionality reduction column"
    )
    credit_card_transactions__Time: float = Field(
        ..., ge=0.00000, description="Number of seconds elapsed between this transaction and the first transaction"
    )
    credit_card_transactions__Amount: float = Field(
        ..., ge=0.00000, description="Transaction amount"
    )


class FraudDetection(BaseModel):
    transaction: float = Field(
        ...,
        const=0,
        const=1,
        description="fraud detection: 0 (fraud) to 1 (legitimate)",
    )
