from crewai import Task
from agents import Nurse, Diagnostic, Treatment_Recommender, Pharm, Nutritionist, Health_Coach
from agents import Health_Expert, Health_Educate, Documentation
from tools import prompt_tool, profile_tool

"""### Tasks

Task: Symptom Collection
"""

Symptom_Collect = Task(
    description=(
        "Given the patients input: {input}, this agent interacts with patients to collect detailed information about their symptoms,"
        "medical history, and any other relevant health details. It uses conversational techniques to gather comprehensive data,"
        "ensuring all aspects of the patient’s condition are recorded."
        "Saves the comprehensive data gotten from the patient"
    ),
    expected_output="The output consists of a detailed patient profile, which includes demographic information such as name, age, gender, e.t.c."
    "It provides a symptom report that captures the nature of the symptoms, including their onset, duration, severity,"
    "and associated factors. The output also includes a medical history section with past conditions, surgeries, allergies,"
    "and current medications. Additionally, it records lifestyle factors such as diet, exercise, and sleep patterns."
    "Any other relevant details provided by the patient are also included."
    # "Use this pdf tools=[symptom_search_tool] and see possibly questions you can ask based on different symptoms"
    "Saves the detailed patient profile",
    # agent=Nurse,
    agent=Nurse,
    tools=[prompt_tool, profile_tool],
    # tools=[prompt_tool, file_save_tool],
    # output_file='outputs/patient_profile.txt',
)

"""Task: Diagnostic_Analysis"""

Diagnostic_Analysis = Task(
    description=(
        "Given the nurse's reported symptoms: {symptoms} and medical data gotten from the Symptom Collection Specialist i.e the Nurse agent, this agent analyzes the information using advanced diagnostic algorithms."
        "It evaluates the symptoms, considers potential underlying causes, and cross-references with medical knowledge to suggest possible"
        "diagnoses. The task is to provide a differential diagnosis that outlines several possible conditions, ranked by likelihood. This agent works hand in hand with the Nurse agent"
    ),
    expected_output=(
        "The output is a detailed diagnostic report that includes a list of possible diagnoses, each accompanied by a brief explanation."
        "The report should prioritize the diagnoses based on likelihood, considering factors such as symptom severity, patient history,"
        "and relevant risk factors. The output also includes recommendations for further tests or examinations that could confirm"
        "or rule out the diagnoses. Additionally, the report provides a rationale for each suggested diagnosis, explaining how the symptoms"
        "and patient data support the conclusion. Explain the diagnosis to the patient and why the patient seems to be having those symptioms"
    ),
    agent=Diagnostic,
    # tools=[read_profile_tool]
)

"""Task: Treatment_Plan_Recommendation"""

Treatment_Plan = Task(
    description=(
        "Given the diagnosed conditions from the Diagnostic Specialist, the detailed patient profile and symptom from the Nurse Agent, in addition with the patient's input,"
        "this agent formulates a comprehensive treatment plan. The agent considers the patient's diagnosis, medical history, and any relevant"
        "lifestyle factors to recommend medications, therapies, and other treatment options. This agent works hand in hand with the Nurse and Diagnostic specialist agent."
    ),
    expected_output=(
        "The output is a personalized treatment plan that includes recommended medications, dosages, and schedules, as well as any necessary"
        "therapies or procedures. The plan should also suggest lifestyle modifications that could improve the patient's health outcomes."
        "Each recommendation should be supported by a rationale that explains why it is suitable for the patient, given their diagnosis"
        "and profile. The treatment plan should be clear and actionable, providing step-by-step guidance for both the patient and healthcare"
        "providers."
    ),
    agent=Treatment_Recommender,
)

"""Task: Pharmacologist"""

Pharma = Task(
    description=(
        "This agent receives the treatment plan from the Diagnostic and Treatment Specialist and recommends specific medications."
        "It takes into account the patient's symptoms, diagnosis, and any potential drug interactions"
        "to provide safe and effective medication options."
    ),
    expected_output="**: The agent provides a detailed list of recommended medications, including their names, dosages, and routes of administration. It includes precise dosage instructions, outlining how often and for how long each medication should be taken. The output also includes information on potential side effects and instructions on what to do if they occur. It provides warnings about possible drug interactions with other medications the patient is taking or conditions they have. Any special instructions or considerations related to the medications are also included, ensuring that both patients and healthcare providers have comprehensive and actionable information",
    agent=Pharm,
    # tools=[agent_question],
)

"""Task: Nutrition Advisor"""

Nutrition = Task(
    description=(
        "This agent receives the symptom report and treatment plan from the Diagnostic and Treatment Specialist. It is responsible for creating personalized dietary recommendations and meal plans based on the patient’s symptoms, health conditions, and overall treatment goals."
    ),
    expected_output="The agent provides a tailored meal plan that includes specific dietary recommendations, such as food types, portion sizes, and meal timings, aligned with the patient's health conditions and treatment goals. It also includes guidelines for balanced nutrition, highlighting key nutrients and dietary adjustments needed to support the patient's health. The output features recipes or meal ideas that adhere to the recommended diet and any necessary adjustments for allergies or dietary restrictions. Additionally, the agent offers tips for incorporating these dietary changes into daily life, including strategies for managing eating habits and maintaining nutritional balance.",
    agent=Nutritionist,
    # tools=[agent_question],
)

"""Health and Wellness Advisor"""

Health_Well_Advisor = Task(
    description=(
        "This agent uses the diagnostic information and treatment plans from the Diagnostic and Treatment Specialist and the preventive measures from the Preventive Health Advisor to provide comprehensive guidance on overall health and wellness. It focuses on enhancing the patient’s well-being through lifestyle recommendations, stress management, and other wellness strategies."
    ),
    expected_output="The agent provides a detailed wellness plan that includes recommendations for lifestyle changes such as physical activity, stress management techniques, and sleep improvement strategies. It offers advice on maintaining mental and emotional health, including coping mechanisms and relaxation exercises. The output also includes suggestions for integrating these wellness practices into daily routines, with actionable steps for improving overall quality of life. Additionally, it provides motivational tips and resources for ongoing support, such as links to wellness programs or apps that can help the patient track their progress and stay engaged with their health goals.",
    agent=Health_Coach,
    # tools=[agent_question],
)

"""Task: Preventive Health Advisor"""

Preventive_Advisor = Task(
    description=(
        "This agent uses the diagnosis and treatment recommendations from the Diagnostic and Treatment Specialist with the Medication Advisor to provide guidance on preventive measures, screenings, and vaccinations. It aims to help patients avoid future health issues through proactive care strategies."
    ),
    expected_output="The output includes personalized recommendations for preventive measures, such as lifestyle changes or new health habits. It provides suggestions for specific tests or screenings based on the diagnosis and risk factors. The agent also recommends vaccinations, including the types, schedules, and any relevant considerations. Furthermore, it outlines a health maintenance plan that details ongoing preventive care activities and monitoring.",
    agent=Health_Expert,
    # tools=[agent_question],
)

"""Task: Medical Research"""

Health_Education = Task(
    description=(
        "Given the patient’s health conditions, treatment plans, and self-care needs gotten from the nurse, the Diagnostic and Treatment Specialist,"
        "this agent provides customized educational content. It selects the most relevant resources from a wide range of formats,"
        "including interactive modules, videos, and articles, to help the patient understand their health situation and how to manage it."
    ),
    expected_output=(
        "The output is a tailored set of educational materials that includes videos, interactive modules, and articles. Each piece of content"
        "is specifically chosen to address the patient's diagnosed conditions, prescribed treatment plans, and recommended self-care practices."
        "The educational materials should be clear, engaging, and easy to understand, with a focus on empowering the patient to take an active"
        "role in managing their health. The output should also include any relevant tips or best practices for the patient to follow."
    ),
    agent=Health_Educate,
)

"""Task: Documentation"""

Document = Task(
    description=("The Documentation and Verification Agent listens to and records all interactions between the patient and the other agents i.e the Symptom Collection Specialist, Diagonistic Specialist, Treatment Specialist, Medication Advisor, Personalized Nutrition Advisor, Health and Wellness Advisor, Preventive Health Advisor, Health Educator."
                 "It organizes this information into a clear, patient-friendly document that highlights key points, such as critical diagnoses,"
                 "treatment plans, medication instructions, lifestyle advice, and follow-up recommendations.This agent work hand in hand with everyother agent to collect detailed report"
                 "Simultaneously, the agent runs a background check on all advice given by the agents to ensure it is accurate, evidence-based,"
                  "and aligns with standard medical ethics and guidelines. This verification process is internal and not shared with the patient,"
                  "but it ensures the highest quality of care is maintained."
    ),

    expected_output="The expected output includes a comprehensive summary document for the patient, capturing all key interactions with the Symptom Collection Specialist, Diagonistic Specialist, Treatment Specialist, Medication Advisor, Personalized Nutrition Advisor, Health and Wellness Advisor, Preventive Health Advisor, Health Educator."
    "This document outlines the primary symptoms, diagnoses, treatments, medications, and any lifestyle or wellness advice provided, "
    "organized in a patient-friendly format. Additionally, the agent produces an internal report that verifies the accuracy of the information "
    "provided by the other agents, checks adherence to medical guidelines, and notes any discrepancies or areas for improvement. "
    "This internal report is kept confidential and used to maintain the integrity and ethical standards of the healthcare provided.",
    agent=Documentation,
)