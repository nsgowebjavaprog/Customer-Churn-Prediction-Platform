"""
Pydantic v2 schemas for the Customer Churn Prediction API.

Responsibilities:
- Validate incoming customer data
- Generate Swagger/OpenAPI documentation
- Define prediction responses
- Define prediction-history CRUD schemas
- Define batch prediction responses
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# =========================================================
# ENUMS
# =========================================================


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"


class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    none_ = "No"


class SupportEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer"
    credit_card = "Credit card"


# =========================================================
# CUSTOMER FEATURES
# =========================================================


class CustomerFeatures(BaseModel):
    """
    Input schema for POST /predict.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "gender": "Female",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure_months": 5,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 89.5,
                "total_charges": 450.0,
                "num_support_calls": 3,
            }
        },
    )

    gender: GenderEnum

    senior_citizen: int = Field(
        ge=0,
        le=1,
        description="0 = No, 1 = Yes",
    )

    partner: YesNoEnum

    dependents: YesNoEnum

    tenure_months: int = Field(
        ge=0,
        le=100,
        description="Customer tenure in months",
    )

    contract: ContractEnum

    internet_service: InternetServiceEnum

    online_security: SupportEnum

    tech_support: SupportEnum

    streaming_tv: SupportEnum

    paperless_billing: YesNoEnum

    payment_method: PaymentMethodEnum

    monthly_charges: float = Field(
        ge=0,
        le=1000,
        description="Monthly charges",
    )

    total_charges: float = Field(
        ge=0,
        description="Total charges",
    )

    num_support_calls: int = Field(
        ge=0,
        le=50,
        description="Number of support calls",
    )


# =========================================================
# PREDICTION RESPONSE
# =========================================================


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float = Field(ge=0, le=1)
    risk_level: str
    model_used: str


# =========================================================
# PREDICTION HISTORY
# =========================================================


class PredictionRecordBase(BaseModel):
    customer_id: str | None = None
    churn_prediction: str
    churn_probability: float = Field(ge=0, le=1)
    risk_level: str
    model_used: str


class PredictionRecordCreate(PredictionRecordBase):
    """
    Schema used when creating a prediction-history record.
    """

    input_payload: dict


class PredictionRecordUpdate(BaseModel):
    """
    Schema used for PATCH requests.

    All fields are optional because PATCH supports
    partial updates.
    """

    customer_id: str | None = None

    churn_prediction: str | None = None

    churn_probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    risk_level: str | None = None

    notes: str | None = None


class PredictionRecordResponse(PredictionRecordBase):
    """
    Schema returned when retrieving prediction history.
    """

    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    created_at: datetime

    notes: str | None = None


class PaginatedHistory(BaseModel):
    """
    Paginated prediction-history response.
    """

    total: int

    page: int = Field(
        ge=1
    )

    page_size: int = Field(
        ge=1
    )

    items: list[PredictionRecordResponse]


# =========================================================
# BATCH PREDICTION
# =========================================================


class BatchPredictionSummary(BaseModel):
    """
    Summary returned after batch CSV prediction.
    """

    total_rows: int = Field(
        ge=0
    )

    valid_rows: int = Field(
        ge=0
    )

    invalid_rows: int = Field(
        ge=0
    )

    churn_count: int = Field(
        ge=0
    )

    errors: list[str] = Field(
        default_factory=list
    )

    download_ready: bool = False










"""
schemas.py
----------
All Pydantic v2 models. These give us:
  - Automatic request validation (type checking, enums, ranges)
  - Automatic OpenAPI / Swagger docs
  - A clean separation between "what the API accepts" (Create/Update)
    and "what the API returns" (Response) -- classic CRUD schema pattern.
"""

'''
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------- Enums (constrain categorical inputs, self-documenting) ----------

class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"


class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    none_ = "No"


class SupportEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer"
    credit_card = "Credit card"


# ---------- Prediction request / response ----------

class CustomerFeatures(BaseModel):
    """Single-customer input payload for POST /predict"""
    model_config = ConfigDict(use_enum_values=True, json_schema_extra={
        "example": {
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "Yes",
            "dependents": "No",
            "tenure_months": 5,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "monthly_charges": 89.5,
            "total_charges": 450.0,
            "num_support_calls": 3,
        }
    })

    gender: GenderEnum
    senior_citizen: int = Field(ge=0, le=1, description="0 = No, 1 = Yes")
    partner: YesNoEnum
    dependents: YesNoEnum
    tenure_months: int = Field(ge=0, le=100)
    contract: ContractEnum
    internet_service: InternetServiceEnum
    online_security: SupportEnum
    tech_support: SupportEnum
    streaming_tv: SupportEnum
    paperless_billing: YesNoEnum
    payment_method: PaymentMethodEnum
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float = Field(ge=0)
    num_support_calls: int = Field(ge=0, le=50)


class PredictionResponse(BaseModel):
    churn_prediction: str          # "Yes" / "No"
    churn_probability: float = Field(description="Probability customer churns (0-1)")
    risk_level: str                # Low / Medium / High
    model_used: str


# ---------- CRUD schemas for prediction history ----------

class PredictionRecordBase(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: str
    churn_probability: float
    risk_level: str
    model_used: str


class PredictionRecordCreate(PredictionRecordBase):
    input_payload: dict


class PredictionRecordUpdate(BaseModel):
    """Used for PATCH — every field optional (partial update)."""
    customer_id: Optional[str] = None
    churn_prediction: Optional[str] = None
    churn_probability: Optional[float] = None
    risk_level: Optional[str] = None
    notes: Optional[str] = None


class PredictionRecordResponse(PredictionRecordBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    notes: Optional[str] = None


class PaginatedHistory(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[PredictionRecordResponse]


# ---------- Batch upload ----------

class BatchPredictionSummary(BaseModel):
    total_rows: int
    valid_rows: int
    invalid_rows: int
    churn_count: int
    errors: list[str] = []
    download_ready: bool
'''