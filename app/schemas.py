from datetime import datetime

from pydantic import BaseModel, EmailStr, constr, Field
from typing import List, Optional, Literal, Dict, Any


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
    shareable_id: Optional[str] = None


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


class ShareableIdUpdate(BaseModel):
    shareable_id: constr(min_length=3, max_length=32)  # type: ignore


class BikeCreate(BaseModel):
    name: str
    brand: str
    model_year: Optional[int] = None
    bike_size: Optional[str] = None


class PointCoord(BaseModel):
    """Position of a point at one kinematics step (index is implicit)."""
    x: float
    y: float


class BikePoint(BaseModel):
    id: str
    type: str
    x: float
    y: float
    name: Optional[str] = None
    coords: List[PointCoord] = Field(default_factory=list)

class PerspectivePoint(BaseModel):
    id: str
    type: str
    x: float
    y: float


class RimEllipse(BaseModel):
    cx: float
    cy: float
    rx: float
    ry: float
    angle_deg: float

class DisplayControlPoint(BaseModel):
    t: float
    offset: float

class DisplayGeometryPoint(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    point_id: Optional[str] = None
    local_t: Optional[float] = None
    local_n: Optional[float] = None


class BodyDisplayGeometry(BaseModel):
    version: int = 1
    interpolation: Literal["polyline", "spline", "pchip"] = "polyline"
    anchor_point_ids: List[str] = Field(default_factory=list)
    points: List[DisplayGeometryPoint] = Field(default_factory=list)
    control_points: List[DisplayControlPoint] = Field(default_factory=list)


class RigidBody(BaseModel):
    id: str
    name: Optional[str] = None
    point_ids: List[str] = Field(default_factory=list)
    type: Optional[str] = None          # "bar" | "shock" | ...
    brake_caliper_point_id: Optional[str] = None  # optional rear-brake caliper anchor point id
    closed: bool = False                # for loops (probably False for linkages)
    length0: Optional[float] = None     # shock eye-to-eye at zero stroke [px or mm]
    stroke: Optional[float] = None      # total shock stroke [same units as length0]
    display_geometry: Optional[BodyDisplayGeometry] = None

class PerspectiveCorrection(BaseModel):
    rear_rim_pts: List[PointCoord] = Field(default_factory=list)
    front_rim_pts: List[PointCoord] = Field(default_factory=list)
    H: List[List[float]] = Field(default_factory=list)       # image -> rectified
    H_inv: List[List[float]] = Field(default_factory=list)   # rectified -> image
    status: Optional[str] = None                             # ok | warning | invalid
    version: int = 1

ScaleSource = Literal["rear_center", "front_center", "wheelbase", "shock_eye"]
PerspectiveMode = Literal["off", "front", "rear", "both_ls"]
ShockType = Literal["air", "coil"]
BikeVisibility = Literal["private", "public"]


class ShockModel(BaseModel):
    coil_rate_n_per_mm: float = 70.0
    coil_preload_n: float = 0.0
    air_chamber_diameter_mm: float = 42.0
    air_chamber_length_mm: float = 95.0
    air_negative_chamber_length_mm: float = 35.0
    air_piston_head_thickness_mm: float = 5.0
    air_shaft_diameter_mm: float = 12.0
    air_initial_pressure_psi: float = 170.0
    air_reference_temp_c: float = 20.0
    air_cold_temp_c: float = 5.0
    air_hot_temp_c: float = 45.0


class ShockPresetOut(BaseModel):
    id: str
    preset_id: str
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    shock_type: ShockType = "air"
    shock_model: ShockModel


class BikeGeometry(BaseModel):
    rear_center_mm: float | None = None
    front_center_mm: float | None = None
    wheelbase_mm: float | None = None
    shock_eye_mm: float | None = None
    scale_mm_per_px: float | None = None
    scale_source: ScaleSource | None = None  # which measurement set the scale
    perspective: PerspectiveCorrection | None = None
    shock_type: ShockType | None = None
    shock_model: ShockModel | None = None
    shock_preset_id: str | None = None


class BikePointsUpdate(BaseModel):
    points: List[BikePoint] = Field(default_factory=list)
    bodies: List[RigidBody] = Field(default_factory=list)


class BikeBodiesOut(BaseModel):
    bodies: List[RigidBody] = Field(default_factory=list)


class BikeBodiesUpdate(BaseModel):
    bodies: List[RigidBody] = Field(default_factory=list)


class KinematicsPoint(BaseModel):
    point_id: str
    x: float
    y: float


class KinematicsStep(BaseModel):
    step_index: int
    shock_stroke: float
    shock_length: float
    rear_travel: Optional[float] = None
    leverage_ratio: Optional[float] = None
    anti_squat: Optional[float] = None
    anti_rise: Optional[float] = None
    shock_spring_rate: Optional[float] = None
    rear_wheel_force: Optional[float] = None


class BikeKinematics(BaseModel):
    rear_axle_point_id: Optional[str] = None
    n_steps: int = 0
    driver_stroke: Optional[float] = None
    steps: List[KinematicsStep] = Field(default_factory=list)
    scaled_outputs: Optional[Dict[str, Any]] = None


class BikeOut(BaseModel):
    id: str
    name: str
    brand: str
    model_year: Optional[int] = None
    bike_size: Optional[str] = None
    user_id: str
    owner_user_id: str
    creator_shareable_id: Optional[str] = None
    can_edit: bool = False
    max_rear_travel_mm: Optional[int] = None
    visibility: BikeVisibility = "private"
    is_verified: bool = False
    verified_by_user_id: Optional[str] = None
    verified_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    hero_media_id: Optional[str] = None
    hero_url: Optional[str] = None
    hero_thumb_url: Optional[str] = None
    hero_perspective_ellipses: Optional[Dict[str, RimEllipse]] = None
    hero_perspective_homography: Optional[dict] = None
    hero_detection_boxes: Optional[dict] = None
    points: Optional[List[BikePoint]] = None
    bodies: Optional[List[RigidBody]] = None
    geometry: BikeGeometry | None = None
    kinematics: Optional[BikeKinematics] = None


class BikeUpdate(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    model_year: Optional[int] = None
    bike_size: Optional[str] = None


class BikeAccessUpdate(BaseModel):
    visibility: Optional[BikeVisibility] = None
    is_verified: Optional[bool] = None


class BikePageSettingsPayload(BaseModel):
    settings: Dict[str, Any] = Field(default_factory=dict)


class BikePageSettingsOut(BikePageSettingsPayload):
    bike_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    
