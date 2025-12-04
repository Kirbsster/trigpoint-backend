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

class BikePoint(BaseModel):
    id: str
    type: str
    x: float
    y: float
    name: Optional[str] = None

class RigidBody(BaseModel):
    id: str
    name: Optional[str] = None
    point_ids: List[str] = Field(default_factory=list)
    closed: bool = False  # for loops (probably False for linkages)

class BikePointsUpdate(BaseModel):
    points: List[BikePoint] = Field(default_factory=list)
    bodies: List[RigidBody] = Field(default_factory=list)

class BikeBodiesOut(BaseModel):
    bodies: List[RigidBody] = Field(default_factory=list)


class BikeBodiesUpdate(BaseModel):
    bodies: List[RigidBody] = Field(default_factory=list)