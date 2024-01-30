# import sqlalchemy
# from sqlalchemy import create_engine

# DATABASE_URL = "postgresql://postgres:ElonMusk123@physician-ai.cbtexq15uzag.us-east-1.rds.amazonaws.com/postgres"

# # Create the engine object
# engine = create_engine(DATABASE_URL)

# #creating a Session class for easier access
# SessionLocal = sqlalchemy.orm.sessionmaker(autocommit=False, autoflush=False, bind=engine)

# #creates a new session
# def get_db():
#     try:
#         db = SessionLocal()
#         yield db
#     finally:
#         db.close()
