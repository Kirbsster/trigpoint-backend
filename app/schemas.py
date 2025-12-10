from datetime import datetime

from pydantic import BaseModel, EmailStr, constr, Field
from typing import List, Optional


# Auth payloads
class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    sub: str
    role: str
    typ: str   # "access" | "refresh"

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class RegisterIn(BaseModel):
    email: EmailStr
    password: constr(min_length=8, max_length=256) # type: ignore

# Users
class UserOut(BaseModel):
    email: EmailStr
    role: str
    is_active: bool

class ForgotPasswordIn(BaseModel):
    email: EmailStr

class ResetPasswordIn(BaseModel):
    token: str
    new_password: constr(min_length=8, max_length=256)

class ChangePasswordIn(BaseModel):
    current_password: constr(min_length=8, max_length=256)
    new_password: constr(min_length=8, max_length=256)

class RegisterOut(UserOut):
    verify_token_dev_only: Optional[str] = None
    verify_link_dev_only: Optional[str] = None

class BikeCreate(BaseModel):
    name: str
    brand: str
    model_year: Optional[int] = None

class BikePoint(BaseModel):
    id: str
    type: str
    x: float
    y: float
    name: Optional[str] = None

# class RigidBody(BaseModel):
#     id: str
#     name: Optional[str] = None
#     point_ids: List[str] = Field(default_factory=list)
#     type: Optional[str] = None
#     closed: bool = False  # for loops (probably False for linkages)
class RigidBody(BaseModel):
    id: str
    name: Optional[str] = None
    point_ids: List[str] = Field(default_factory=list)
    type: Optional[str] = None          # "bar" | "shock" | ...
    closed: bool = False                # for loops (probably False for linkages)

    # NEW: shock-specific geometry (optional for non-shock bodies)
    length0: Optional[float] = None     # shock eye-to-eye at zero stroke [px or mm]
    stroke: Optional[float] = None      # total shock stroke [same units as length0]
    
class BikeGeometry(BaseModel):
    rear_center_mm: float | None = None
    scale_mm_per_px: float | None = None  # or px_per_mm if you prefer

class BikeOut(BaseModel):
    id: str
    name: str
    brand: str
    model_year: Optional[int] = None
    user_id: str
    created_at: datetime
    updated_at: datetime
    hero_media_id: Optional[str] = None
    hero_url: Optional[str] = None
    points: Optional[List[BikePoint]] = None
    bodies: Optional[List[RigidBody]] = None
    geometry: BikeGeometry | None = None

class BikePointsUpdate(BaseModel):
    points: List[BikePoint] = Field(default_factory=list)
    bodies: List[RigidBody] = Field(default_factory=list)

class BikeBodiesOut(BaseModel):
    bodies: List[RigidBody] = Field(default_factory=list)

class BikeBodiesUpdate(BaseModel):
    bodies: List[RigidBody] = Field(default_factory=list)

class RearCenterUpdate(BaseModel):
    rear_center_mm: float = Field(gt=0, description="Rear centre in mm")
    


