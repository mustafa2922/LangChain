from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import Annotated, Optional
from pydantic import BaseModel, Field, EmailStr
from datetime import date

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Schema

class Introduction(BaseModel):
    name: str = Field(description='name of the student')
    email: EmailStr = Field(description='email of the student')
    institute: str = Field(description="student's institutes")
    degree: str = Field(description='degree of student')
    year_of_study: int = Field(gt=0, lt=5 , description='Current year of study')
    gpa: float = Field(gt=0, lt=5 , description='gpa score of student')
    student_id: str = Field(description="student's id")
    age: int = Field(gt=12, lt=23 , description='age of the student')
    dob: date = Field(description='date of birth in format: YYYY-MM-DD')
    phone_number: str = Field(description='phone number of the student' , pattern=r"^\+?[1-9]\d{1,14}$" )
    city: str = Field(description='city of the student')
    country: str = Field(description='name of the country')
    languages: list[str] = Field(description='spoken languages in a list')
    certifications: list[str] = Field(description='certifications of student in a list')
    projects_completed: int =  Field(description='projects completed by student')
    hobbies: list[str] =  Field(description='hobbies of student')
    relationship_status: str =  Field(description='marital status of student')
    github_username: str = Field(description='github profile', default=None)
    linkedin_url: str = Field(description='linkedin profile', default=None)
    internship_experience: int = Field(description='past experience of student' , gt=0)
    career_goal: Optional[list[str]] =  Field(description='carrer goals of student' , default=None)

structured_model = model.with_structured_output(Introduction.model_json_schema())

result = structured_model.invoke("""A 20-year-old boy named Ahmed Raza lives in Karachi, Pakistan. He studies Software Engineering at the University of Karachi, where he is currently in his third year with a 3.5 GPA. His student ID is SE2023012, and his official email address is ahmed.raza@example.com.He has completed 10 programming projects, is fluent in English and Urdu, and holds a Java certification. He enjoys football and reading books in his free time. Ahmed’s phone number is +92-300-1234567, his date of birth is 2005-03-15, and he is single. His GitHub username is ahmedrazadev, and his LinkedIn profile is linkedin.com/in/ahmedrazadev. He has 1 internship experience at a local tech company and aims to become a backend engineer after graduation.""")


print(type(result))