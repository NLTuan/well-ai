"""### Using Crew AI Tools"""

from crewai_tools import BaseTool
from crewai_tools import PDFSearchTool
from crewai_tools import RagTool

"""Patient Prompting"""

class PromptingPatientTool(BaseTool):
    name: str = "Patient Prompting"
    description: str = "When more information is needed from the patient, this tool is used to ask them a question directly."

    def _run(self, question: str) -> str:
        # Your tool's logic here
        print(f"Question: {question}")
        answer = input()
        return answer

prompt_tool = PromptingPatientTool()

rag_tool = RagTool(config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama3.1",
            ),
        ),

        embedder=dict(
            provider="ollama",
            config=dict(
                model="nomic-embed-text",
            ),
        ),
    )
)

symptom_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/Symptom-collection.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

diagonise_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/Oxford-Handbook-of-Clinical-Diagnosis.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

pain_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/oxford-handbook-of-pain-management.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

treatment_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/Oxford-Handbook-of-Practical-Drug-Therapy.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

medication_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/OXFORD-HANDBOOK-OF-CLINICAL-PHARMACY.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

nutrition_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/Oxford-Handbook-of-Nutrition-and-Dietetics.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

health_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/Health-and-Wellness.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

preventive_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/Guidelines-for-preventive-activities.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

educate_search_tool = PDFSearchTool(
    pdf="/notebooks/Docs/Oxford-Handbook-of-Public-Health-Practice.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.1")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

import json
from typing import Dict, Any, Optional

# Assuming BaseTool definition is available
class ProfileTool(BaseTool):
    name: str = "Patient Profile"
    description: str = "Handles saving and updating patient profiles in JSON format."

    def _run(self, operation: str, patient_id: Optional[str] = None, updated_info: Optional[Dict[str, Any]] = None, file_path: Optional[str] = "patient_profile.json") -> str:
        if not patient_id or not updated_info:
            return "Invalid operation or missing parameters."

        # Load existing profile if it exists
        try:
            with open(file_path, 'r') as f:
                profile = json.load(f)
        except FileNotFoundError:
            profile = {}

        # Ensure the profile includes all relevant fields
        required_fields = {
            "patient_id": "",
            "symptoms": "",
            "medical_history": {
                "current_medications": [],
                "lifestyle_factors": {},
                "allergies": [],
                "recent_lab_results": []
            },
            "diagnoses": [],
            "treatment_plan": {},
            "trigger": {}
        }

        # Merge required fields with the provided updated_info
        updated_info = {**required_fields, **updated_info}

        if operation == "save":
            # Save or overwrite the profile information
            profile[patient_id] = updated_info
            message = f"Profile for patient {patient_id} saved successfully."

        elif operation == "update":
            # Update the existing profile or create a new one if it doesn't exist
            if patient_id in profile:
                profile[patient_id].update(updated_info)
                message = f"Profile for patient {patient_id} updated successfully."
            else:
                profile[patient_id] = updated_info
                message = f"Profile for patient {patient_id} did not exist and has been created."

        else:
            return "Invalid operation."

        # Write the profile back to the JSON file
        try:
            with open(file_path, 'w') as f:
                json.dump(profile, f, indent=4)
        except IOError as e:
            return f"Failed to save the profile: {str(e)}"

        return message

# Example usage:
profile_tool = ProfileTool()


class ReadProfileTool(BaseTool):
    name: str = "Read Patient Profile"
    description: str = "Reads and returns the contents of a patient profile from a specified JSON file."

    def _run(self, file_path: Optional[str] = "patient_profile.json") -> str:
        try:
            with open(file_path, 'r') as f:
                profile = json.load(f)
            return json.dumps(profile, indent=4)  # Return the content as a formatted JSON string

        except FileNotFoundError:
            return f"File '{file_path}' not found."

        except json.JSONDecodeError:
            return f"File '{file_path}' contains invalid JSON."

        except IOError as e:
            return f"Failed to read the file '{file_path}': {str(e)}"

# Example usage:
read_profile_tool = ReadProfileTool()

# Reading the contents of the patient profile JSON file
# result = read_profile_tool._run(file_path="patient_profile.json")
# print(result)

