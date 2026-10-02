from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNoUnknown = Literal["yes", "no", "unknown"]
Month = Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


class ClientProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1, max_length=64, description="Pseudonymized identifier, never the name")
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown
    contact: Literal["cellular", "telephone"]
    month: Month
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    campaign: int = Field(ge=1, le=100, description="Number of contacts in this campaign, including this one")
    pdays: int = Field(ge=0, le=999, description="Days since the last contact; 999 = never contacted")
    previous: int = Field(ge=0, le=100)
    poutcome: Literal["failure", "nonexistent", "success"]
    emp_var_rate: float = Field(ge=-10, le=10)
    cons_price_idx: float = Field(ge=50, le=150)
    cons_conf_idx: float = Field(ge=-100, le=100)
    euribor3m: float = Field(ge=-1, le=20)
    nr_employed: float = Field(ge=1000, le=10000)


class Prediction(BaseModel):
    client_id: str
    probability: float
    decision: Literal["call", "do_not_call", "advisor_review"]
    threshold: float
    model_version: str


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1, max_length=64)
    outcome: Literal["subscribed", "declined", "unreachable", "opted_out"]
