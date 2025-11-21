from pydantic import BaseModel, EmailStr

class UserIn(BaseModel):
    email: EmailStr
    password: str
    role: str = "user"
    is_active: bool = True

class UserOut(BaseModel):
    email: EmailStr
    role: str
    is_active: bool