from pydantic import BaseModel, Field, EmailStr
from typing import Optional

# Student model: ek simple schema define kiya hai
class Student(BaseModel):
    name: str
    age: Optional[int] = None
    cgpa: float = Field(gt=0, lt=10, default=5, description="This show performance of a student")
    email: EmailStr  # email ko valid format me check karta hai

# Data dictionary: yahan values hai
new_student = {'name': 'saurabh', 'age': '20', 'cgpa': 8.2, 'email': 'abc@gmil.com'}

# ** unpack karta hai dictionary ko keyword arguments me
student = Student(**new_student)
student_json = student.model_dump_json()  # object ko JSON format me convert karta hai
print(student.age)  # age print kar raha hai
