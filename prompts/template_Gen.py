from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template = """ 
You are an AI knowledge assistant.  
The user will give you three inputs:  
1. Country name = {country}  
2. Perspective = {perspective}  
3. Writing mode = {mode}  

Your task is to:  
- Focus on the given country.  
- Explain it through the requested perspective.  
- Present the answer strictly in the chosen writing mode.  

If the perspective or mode is unclear, politely ask the user for clarification before answering.  
Make sure the output is accurate, concise, and well-structured according to the chosen mode.
    """,
   input_variables = ["country","perspective","mode"],
   validate_template = True
)

template.save('./template.json')