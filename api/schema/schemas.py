from pydantic import BaseModel, Field



class HealthCheckResult(BaseModel):
    success: bool

class PredictionResult(BaseModel):
    probability_of_survival: float = Field(
        ge=0, le=1, description="Probability of survival"
    )
