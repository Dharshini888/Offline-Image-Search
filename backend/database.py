from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, LargeBinary

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import os
import logging
from sqlalchemy import text

Base = declarative_base()
# ensure the database path is absolute and always points to the workspace data folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "db.sqlite"))
DB_URL = f"sqlite:///{DB_PATH}"

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)
    original_path = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Metadata
    make = Column(String)
    model = Column(String)
    lat = Column(Float)
    lon = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    size_bytes = Column(Integer)
    # average color (RGB 0-255)
    avg_r = Column(Float, default=0.0)
    avg_g = Column(Float, default=0.0)
    avg_b = Column(Float, default=0.0)
    
    # Content info
    ocr_text = Column(Text)
    scene_label = Column(String)
    is_duplicate = Column(Boolean, default=False)
    duplicate_of = Column(Integer, ForeignKey('images.id'))
    person_count = Column(Integer, default=0)
    is_favorite = Column(Boolean, default=False)
    
    # Relationships
    faces = relationship("Face", back_populates="image")
    album_id = Column(Integer, ForeignKey('albums.id'))
    album = relationship("Album", back_populates="images")

class Face(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    bbox = Column(String)          # JSON string: [top, right, bottom, left]
    embedding_idx = Column(Integer) # Index in face_vectors.npy (legacy)
    face_embedding = Column(LargeBinary)  # 128-d float32 blob for clustering
    
    person_id = Column(Integer, ForeignKey('people.id'))
    image = relationship("Image", back_populates="faces")
    person = relationship("Person", back_populates="faces")

class Person(Base):
    __tablename__ = 'people'
    id = Column(Integer, primary_key=True)
    name = Column(String, default="Unknown")
    cover_face_id = Column(Integer)
    faces = relationship("Face", back_populates="person")

class Album(Base):
    __tablename__ = 'albums'
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    type = Column(String) # 'trip', 'event', 'manual'
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    cover_image_id = Column(Integer)
    images = relationship("Image", back_populates="album")

# Dependency helpers
from sqlalchemy import create_engine
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # ensure data directory exists
    if not os.path.exists("../data"):
        os.makedirs("../data")

    # create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # perform simple migrations for sqlite (add columns if missing)
    # currently only adding 'is_favorite' to images table
    # make sure the migration is run inside a transaction and any errors are logged
    with engine.begin() as conn:  # this will commit automatically or rollback on error
        try:
            # check if column exists
            res = conn.execute(text("PRAGMA table_info(images)"))
            cols = [row[1] for row in res.fetchall()]
            if "is_favorite" not in cols:
                conn.execute(text("ALTER TABLE images ADD COLUMN is_favorite BOOLEAN DEFAULT 0"))
                logging.getLogger("database").info("Added missing column 'is_favorite' to images table")
            # add average color columns if missing
            for col in ("avg_r","avg_g","avg_b"):
                if col not in cols:
                    conn.execute(text(f"ALTER TABLE images ADD COLUMN {col} FLOAT DEFAULT 0.0"))
                    logging.getLogger("database").info(f"Added missing column '{col}' to images table")
            # ensure scene_label exists (for object tags)
            if "scene_label" not in cols:
                conn.execute(text("ALTER TABLE images ADD COLUMN scene_label STRING"))
                logging.getLogger("database").info("Added missing column 'scene_label' to images table")
        except Exception as e:
            # log migration errors but let init continue
            logging.getLogger("database").error(f"Error during migration: {e}")

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
