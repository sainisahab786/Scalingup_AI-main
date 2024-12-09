from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import fitz  # Importing the correct module from PyMuPDF
from fastapi.middleware.cors import CORSMiddleware
import re
import json
import google.generativeai as genai
import pandas as pd
import uvicorn 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to analyze strengths and weaknesses based on custom criteria
def get_top_strengths_weaknesses(answer):
    # Ensure answer is parsed as a list of dictionaries if it's a JSON string
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError:
            raise ValueError("Input data is not valid JSON.")

    # Check if each entry is a dictionary
    if not all(isinstance(entry, dict) for entry in answer):
        raise ValueError("Each entry in the answer must be a dictionary.")

    # Dictionary to store results in the specified format
    results = {"Strength_and_Weaknesses": {}}

    # Group data by "Main Section"
    sections = {}
    for entry in answer:
        main_section = entry.get("Main Section")
        if main_section not in sections:
            sections[main_section] = []
        sections[main_section].append(entry)

    # Analyze strengths and weaknesses for each section
    for main_section, entries in sections.items():
        # Sort entries by "You" and "Team" to identify strengths and weaknesses
        strengths = sorted(
            [entry for entry in entries if entry.get("Strengths/Weaknesses") == "Strength"],
            key=lambda x: (float(x.get("You", 0)) + float(x.get("Team", 0))),
            reverse=True
        )
        weaknesses = sorted(
            [entry for entry in entries if entry.get("Strengths/Weaknesses") == "Focus Area"],
            key=lambda x: (float(x.get("You", 0)) + float(x.get("Team", 0)))
        )

        # Get the top 2 strengths and weaknesses for the current main section
        top_strengths = [entry["Sub Sections"] for entry in strengths[:2]]
        top_weaknesses = [entry["Sub Sections"] for entry in weaknesses[:2]] if weaknesses else ["No Weakness"]

        # Ensure weaknesses list only includes "No Weakness" if no actual weaknesses exist
        if not strengths:
            top_strengths = ["No strengths"]

        if not weaknesses:
            top_weaknesses = ["No Weakness"]

        # Store results in the required format
        results["Strength_and_Weaknesses"][main_section] = {
            "Strengths": top_strengths,
            "Weaknesses": top_weaknesses
        }

    return results["Strength_and_Weaknesses"] 

def custom_prompt(output, question):
    prompt = f"""This is a report for scaling up a business:{output}, and this is the question you need to answer from the report: {question}.
        I want a response from you in which you write the answer of the question asked."""
        
    genai.configure(api_key='AIzaSyB798GofH8tgcotUrXYu1Wf38AA_XTisYM')
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')

    completion = model.generate_content(
        prompt,
        generation_config={
            'temperature': 0,
        }
    )
    answer= completion.text
    
    answer = re.sub(r"[\*]", "", answer)
    answer = re.sub(r"json", "", answer)
    answer = re.sub(r"\\n", " ", answer)
    answer = re.sub(r"JSON", "", answer)
    answer = re.sub(r"```", "", answer)
    return answer

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), question: str = Form("")) -> JSONResponse:
    # Ensure the uploaded file is a PDF
    if not file.filename.endswith('.pdf'):
        return {"error": "File must be a PDF."}

    # Read the PDF file and extract text
    pdf_document = fitz.open(stream=await file.read(), filetype="pdf")
    extracted_text = ""
    
    for page in pdf_document:
        extracted_text += page.get_text()
        
    pdf_document.close()

    # Find all occurrences of "you" with associated values
    pattern = r'(.*?)\byou\s+team\s+peers\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)'

    # Find all matches along with the preceding line for each "you" occurrence
    matches = list(re.finditer(pattern, extracted_text, re.DOTALL | re.MULTILINE))
    
    # Check if no matches were found
    if not matches:
        return JSONResponse(content={"message": "Please check the PDF, as it should contain all the values for 'you', 'team', and 'peers'."})

    # Create a dictionary to store results for each occurrence
    results = {}

    def Strength_and_Weaknesses(you_score, team_score):
    # Calculate variance
        variance = abs(you_score - team_score)

        # Basic classifications when both scores fall in the same category
        if you_score >= 7 and team_score >= 7:
            return "Strength"
        elif you_score < 4 and team_score < 4:
            return "Focus Area"
        elif 4 <= you_score < 7 and 4 <= team_score < 7:
            return "Moderate/Neutral"

        # Mixed conditions
        elif (you_score >= 7 and 4 <= team_score < 7) or (team_score >= 7 and 4 <= you_score < 7):
            # Condition 1: One value is "Strength" and the other is "Moderate"
            if 0 <= variance <= 2:
                return "Strength"
            elif 2 < variance <= 4:
                return "Moderate/Neutral"
            elif 4 < variance <= 6:
                return "Focus Area"
        elif (you_score >= 7 and team_score < 4) or (team_score >= 7 and you_score < 4):
            # Condition 2: One value is "Strength" and the other is "Focus"
            return "Focus Area"
        elif (4 <= you_score < 7 and team_score < 4) or (4 <= team_score < 7 and you_score < 4):
            # Condition 3: One value is "Moderate" and the other is "Focus"
            return "Focus Area" if variance > 3 else "Moderate/Neutral"

        return "Moderate/Neutral"
    

    def variance_category(you_score, team_score):
          # Calculate absolute variance
        variance = abs(you_score - team_score)
        
        # Classify the variance
        if variance > 3:
            return "High Variance"
        elif 0 <= variance <= 3:
            return "Low Variance"   
        
    def variance(you_score, team_score):
          # Calculate absolute variance
        variance = abs(you_score - team_score)
        
        return round(variance, 1)   

    # Process each match
    for i, match in enumerate(matches):
        preceding_line = match.group(1).strip().splitlines()[-1]  # Get the last line before "you"
        
        # Check if the sub-section is 'example' and skip it
        if "example" in preceding_line.lower():
            continue  

        # Store the values in a separate dictionary for each match
        results[f"instance_{i+1}"] = {
            "Sub section": preceding_line,
            "values": {
                "you": float(match.group(2)),
                "team": float(match.group(3)),
                "peers": float(match.group(4))
            }
        }
        
    # Prepare the categorized results
    categorized_results = {}
   
    for instance, data_entry in results.items():
        sub_section = data_entry["Sub section"]
        you_score = data_entry["values"]["you"]
        team_score = data_entry["values"]["team"]
        peer_score = data_entry["values"]["peers"]
        variance_abs = variance(you_score, team_score)
        variance_cat = variance_category(you_score, team_score)
        Strength_Weaknesses = Strength_and_Weaknesses(you_score, team_score)

        categorized_results[instance] = {
            "Sub Section": sub_section,
            "You": you_score,
            "Team": team_score,
            "Variance":variance_abs,
            
            "Variance Category" : variance_cat,
            "Strengths/Weaknesses": Strength_Weaknesses
        }
        
    
        
    # Step 1: First LLM model for the main analysis
    prompt = f"""
        <pdf data>
        {extracted_text}
        </pdf data>
    prompt = 
    "Please analyze the following JSON data{results} and {categorized_results}  that contains the assessment for various sub-sections of our organization. For each sub-section please find the main section of the sub section . 
    example of main section are People,Strategy,execution,finance and cash and you, and make sure to take all the values from the {categorized_results} including the Sub Section, You, Team, Peer, Variance,  Variance Category, Strengths/Weaknesses and not from {extracted_text}.

    Output the results in JSON format, including the following fields please do not change the categorized result
    - "Main Section"
    - "Sub Sections"
    - "You"
    - "Team"
    - "Peer"
    - "Variance"
    - "Variance Category"
    - "Strengths/Weaknesses"
    "
    Give the output in JSON format only in the following markdown:
    JSON
    ...
    """
    # Use LLM to process the question based on the fetched insurance data
    genai.configure(api_key='AIzaSyB798GofH8tgcotUrXYu1Wf38AA_XTisYM')
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')

    completion = model.generate_content(
        prompt,
        generation_config={
            'temperature': 0,
        }
    )

    answer = completion.text
    answer = re.sub(r"[\*]", "", answer)
    answer = re.sub(r"json", "", answer)
    answer = re.sub(r"\\n", " ", answer)
    answer = re.sub(r"JSON", "", answer)
    answer = re.sub(r"```", "", answer)
    
    output = {}
    
    # print(output)
    # output["filename"]= file.filename
    # print("saferhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh", output, "dsfkhgerkjfbwejkgnf3jkrvnjkefr")
    # Step 2: Get top strengths and weaknesses from the second LLM call
    # print("Formatted Code: ",answer)
    strengths_weaknesses = get_top_strengths_weaknesses(answer)
    output={"filename": file.filename, "data": json.loads(answer), "Strength_and_Weaknesses": strengths_weaknesses}
    if question:
        answer=custom_prompt(output, question)
        output['Question']= question
        output['Answer']= answer
        return JSONResponse(content=output)
    else:
        output['Question']= "No custom question"
        output['Answer']= ""
        return JSONResponse(content=output)
    # Return both results as part of the response
    
if __name__=="__main__":
    uvicorn.run("main:app",port=8080, reload=True)