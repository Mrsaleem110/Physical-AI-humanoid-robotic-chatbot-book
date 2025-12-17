from typing import Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from ..models.user import User, BackgroundLevel
from ..config import settings
import uuid

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token configuration
SECRET_KEY = settings.better_auth_secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class AuthService:
    def __init__(self):
        self.pwd_context = pwd_context

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against a hashed password."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a plain password."""
        return pwd_context.hash(password)

    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate a user by email and password."""
        user = db.query(User).filter(User.email == email).first()
        if not user or not self.verify_password(password, user.password_hash):
            return None
        return user

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def get_current_user(self, token: str, db: Session) -> Optional[User]:
        """Get the current user from a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
        except JWTError:
            return None

        user = db.query(User).filter(User.id == user_id).first()
        return user

    def create_user(self, db: Session, email: str, password: str, first_name: str = None,
                   last_name: str = None, software_background: BackgroundLevel = BackgroundLevel.NONE,
                   hardware_background: BackgroundLevel = BackgroundLevel.NONE,
                   robotics_experience: BackgroundLevel = BackgroundLevel.NONE) -> User:
        """Create a new user with hashed password."""
        user_id = str(uuid.uuid4())
        hashed_password = self.get_password_hash(password)

        db_user = User(
            id=user_id,
            email=email,
            password_hash=hashed_password,
            first_name=first_name,
            last_name=last_name,
            software_background=software_background,
            hardware_background=hardware_background,
            robotics_experience=robotics_experience,
            personalization_level=BackgroundLevel.BEGINNER  # Default to beginner
        )

        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    def update_user_profile(self, db: Session, user_id: str, first_name: str = None,
                           last_name: str = None, software_background: BackgroundLevel = None,
                           hardware_background: BackgroundLevel = None,
                           robotics_experience: BackgroundLevel = None,
                           personalization_level: BackgroundLevel = None) -> User:
        """Update user profile information."""
        db_user = db.query(User).filter(User.id == user_id).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Update fields if provided
        if first_name is not None:
            db_user.first_name = first_name
        if last_name is not None:
            db_user.last_name = last_name
        if software_background is not None:
            db_user.software_background = software_background
        if hardware_background is not None:
            db_user.hardware_background = hardware_background
        if robotics_experience is not None:
            db_user.robotics_experience = robotics_experience
        if personalization_level is not None:
            db_user.personalization_level = personalization_level

        db.commit()
        db.refresh(db_user)
        return db_user