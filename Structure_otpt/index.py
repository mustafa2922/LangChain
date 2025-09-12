from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Schema

class Introduction(TypedDict):
    name: Annotated[str, 'name of the student']
    email: Annotated[str, 'email of the student']
    institute: Annotated[str, "student's institutes"]
    degree: Annotated[str, 'degree of student']
    year_of_study: Annotated[int, '1st or 2nd or 3rd or 4th']
    gpa: Annotated[float, 'gpa score of student']
    student_id: Annotated[str, "student's id"]
    age: Annotated[int, 'age of student']
    dob: Annotated[str, 'date of birth in format: YYYY-MM-DD']
    phone_number: Annotated[str, 'phone number with country code']
    city: Annotated[str, 'name of the city']
    country: Annotated[str, 'name of the country']
    languages: Annotated[list[str], 'spoken languages']
    certifications: Annotated[list[str], 'certifications of student']
    projects_completed: Annotated[int, 'projects completed by student']
    hobbies: Annotated[list[str], 'hobbies of student']
    relationship_status: Annotated[str, 'marital status of student']
    github_username: Annotated[str, 'github profile']
    linkedin_url: Annotated[str, 'linkedin profile']
    internship_experience: Annotated[int, 'past experience of student']
    career_goal: Annotated[Optional[list[str]], 'carrer goals of student']

structured_model = model.with_structured_output(Introduction)

result = structured_model.invoke("""A 20-year-old boy named Ahmed Raza lives in Karachi, Pakistan. He studies Software Engineering at the University of Karachi, where he is currently in his third year with a 3.5 GPA. His student ID is SE2023012, and his official email address is ahmed.raza@example.com.He has completed 10 programming projects, is fluent in English and Urdu, and holds a Java certification. He enjoys football and reading books in his free time. Ahmed’s phone number is +92-300-1234567, his date of birth is 2005-03-15, and he is single. His GitHub username is ahmedrazadev, and his LinkedIn profile is linkedin.com/in/ahmedrazadev. He has 1 internship experience at a local tech company and aims to become a backend engineer after graduation.""")


print(result)