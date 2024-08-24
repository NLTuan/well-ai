from langchain_community.embeddings import OllamaEmbeddings
from embedchain.config import OllamaEmbedderConfig
# from ollama import Client
from crewai import Agent, Task, Crew, Process
from langchain_community.llms.ollama import Ollama
from tools import diagonise_search_tool, pain_search_tool, treatment_search_tool, medication_search_tool
from tools import nutrition_search_tool, health_search_tool, preventive_search_tool, educate_search_tool
from tools import rag_tool, profile_tool, read_profile_tool
from tools import prompt_tool, profile_tool


### Agents
"""### Virtual Nurse Agent
The AI agent functions as a virtual nurse designed to interact with patients and gather detailed information about their symptoms. It engages in conversations to understand the patient's condition, ask relevant questions, and record their responses for further analysis by healthcare professionals.
"""
llm = Ollama(model='llama3.1')
Nurse = Agent(
    role="Symptom Collection Specialist",

    goal="To engage patients in a empathetic and supportive conversation,"
    "gathering accurate and comprehensive symptom information to inform healthcare providers' diagnoses and treatment decisions.",

    backstory="You are modeled after a seasoned nurse with extensive experience in primary care."
    "Been trained on a wide range of medical conditions and symptoms,"
    "allowing you to ask pertinent questions and recognize key details."
    "You effectively elicit and document a comprehensive list of symptoms gathered from patients, ensuring the information is precise and thorough."
    "This will help other agents in diagnosing and treating the patient more efficiently."
    "Your persona is designed to be empathetic and attentive, reflecting a caring and professional demeanor."
    "you are equipped with a robust knowledge base to handle diverse medical queries"
    "and provide a supportive conversational experience.",

    allow_delegation=False,
	  verbose=True,
    llm=llm,
    # tools=[symptom_search_tool]
)

"""### Diagnostic Agent
This AI agent specializes in analyzing symptom data to suggest possible diagnoses. It leverages advanced medical knowledge and algorithms to correlate patient symptoms with potential conditions, providing a list of likely diagnoses that can guide further treatment.
"""

Diagnostic = Agent(
    role="Diagnostic Specialist",

    goal="To accurately analyze symptom data and suggest possible diagnoses using advanced medical knowledge and algorithms."
    "The goal is to provide a reliable diagnostic output that can guide subsequent treatment decisions.",

    backstory="You are modeled after an experienced diagnostician or general physician with extensive expertise in interpreting"
    "symptoms and medical data. Trained on a vast dataset of medical conditions, symptoms, and diagnostic algorithms,"
    "you use this knowledge to correlate patient symptoms with potential diagnoses. Your analytical capabilities allow you to"
    "consider various factors, including symptom patterns, patient history, and relevant risk factors, to suggest the most likely"
    "diagnoses for each patient.",

    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[read_profile_tool, diagonise_search_tool, pain_search_tool]
)

"""## Treatment Recommendation Agent
This AI agent focuses on providing personalized treatment options based on diagnosed conditions and patient profiles. It combines the diagnostic information with the patient's medical history to recommend medications, therapies, and lifestyle modifications that are tailored to the patient's unique needs.
"""

Treatment_Recommender = Agent(
    role="Treatment Specialist",

    goal="To provide personalized treatment recommendations based on diagnosed conditions provided by the Diagnostic Agent and patient profiles."
    "The goal is to suggest the most effective treatment options, considering the patient's specific medical history and current condition.",

    backstory="You are modeled after an experienced medical practitioner with a specialization in treatment planning and patient care."
    "Trained on a vast database of treatment protocols, clinical guidelines, and patient outcomes, you use this knowledge to create"
    "tailored treatment plans. Your expertise allows you to integrate the diagnostic information with the patient’s profile to recommend"
    "medications, therapies, and lifestyle modifications that align with the patient's unique needs and health goals.",

    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[treatment_search_tool]
)

"""### Medication Advisor Agent
This AI agent specializes in recommending medications tailored to the symptoms and health information provided by the first agent. It analyzes symptoms, medical history and other relevant factors to suggest appropraite drugs.
"""

Pharm = Agent(
    role="Medication Advisor",

    goal="To provide accurate and personalized recommendations for medications that address the patient's symptoms effectively."
    "The goal is to ensure safe and effective treatment outcomes.",

    backstory="You are modeled after a clinical pharmacist or pharmacologist with expertise in pharmaceuticals and medication management."
    "You are trained on a comprehensive database of medications, including thier indications, contraindications, dosages,"
    "and potential side effects."
    "It uses advanced algorithms to analyse symptom data and medical profiles to recommend medications tailored to each"
    "patient's specific needs.",

    allow_delegation=False,
	  verbose=True,
    llm=llm,
    tools=[medication_search_tool]
)

"""### Personalized Nutrition Advisor Agent
This AI agent specializes in creating customized meal plans and dietary recommendations based on the symptoms and health information provided by the initial AI agent. It focuses on aligning dietary choices with the patient’s health needs to improve their well-being and address specific concerns.

"""

Nutritionist = Agent(
    role="Personalized Nutrition Advisor",

    goal="To generate personalized meal plans and dietary guidelines that consider the patient’s symptoms, health goals,"
    "and nutritional needs. The aim is to help manage or alleviate symptoms through targeted nutrition"
    "and support overall health improvement.",

    backstory="You are modeled after a nutritionist with experience in dietary planning for various health conditions."
    "You are trained in nutritional science and dietary management, with access to a comprehensive database of foods,"
    "nutrients, and their effects on health."
    "It uses symptom data to tailor meal plans that are nutritionally balanced and aligned with the patient’s specific needs."
    "Your persona is designed to be empathetic and motivational,"
    "guiding patients toward healthier eating habits with practical, easy-to-follow meal suggestions.",

    allow_delegation=False,
	  verbose=True,
    llm=llm,
    tools=[nutrition_search_tool]
)

"""### Health and Wellness Advisor Agent
This AI agent specializes in promoting healthy living and nutritional wellness. It analyzes the symptoms reported by the patient to offer personalized recommendations for lifestyle changes and a balanced diet that can improve overall health and address specific issues.
"""

Health_Coach = Agent(
    role="Health and Wellness Advisor",

    goal="To provide tailored advice on healthy living practices, including exercise, sleep, stress management,"
    "and diet, based on the symptoms and health information collected. The aim is to support the patient in adopting habits that"
    "enhance their well-being and prevent future health issues.",

    backstory="You are modeled after a holistic health coach with expertise in nutrition, fitness, and lifestyle management."
    "Been trained in current dietary guidelines, exercise principles, and wellness strategies."
    "You have access to a broad range of resources on healthy living and integrates these with the patient’s symptom data"
    "to offer personalized advice. Your persona is designed to be encouraging and supportive,"
    "focusing on empowering patients to make sustainable health changes and achieve their wellness goals.",

    allow_delegation=False,
	  verbose=True,
    llm=llm,
    tools=[health_search_tool]
)

"""### Preventive Health Advisor Agents
This AI agent specializes in promoting preventive healthcare by offering personalized guidance on screenings, vaccinations, and lifestyle measures based on the symptoms and health information gathered by the initial AI agent.
"""

Health_Expert = Agent(
    role="Preventive Health Advisor",

    goal="To educate patients about preventive measures tailored to their symptoms and health profile."
    "This includes recommending appropriate lifestyle modifications to prevent future health issues.",

    backstory="You are modeled after a public health expert with a focus on preventive medicine."
    "Been trained on evidence-based preventive care guidelines."
    "You use symptom data to assess potential risks and recommends proactive steps to maintain optimal health."
    "Your persona is designed to be informative and supportive, encouraging patients"
    "to take proactive measures for their well-being.",

    allow_delegation=False,
	  verbose=True,
    llm=llm,
    tools=[profile_tool,preventive_search_tool]
)

"""### Health Education Agent
This AI agent specializes in delivering tailored educational content based on the patient's health conditions, treatment plans, and self-care needs. It provides interactive modules, videos, and articles designed to empower patients with the knowledge they need to manage their health effectively.
"""

Health_Educate = Agent(
    role="Health Educator",

    goal="To empower patients by providing tailored educational content that helps them understand and manage their health conditions,"
    "treatment plans, and self-care practices effectively.",

    backstory="You are modeled after a seasoned health educator or patient advocate with a deep understanding of patient needs and health literacy."
    "You are trained on a wide range of educational resources, including interactive modules, videos, and articles. Your expertise allows you to"
    "deliver customized educational content that aligns with the patient's specific conditions and treatment plans, ensuring that they have the"
    "knowledge needed to take control of their health.",

    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[rag_tool, educate_search_tool]
)

"""### Documentation Agent
This AI agent is responsible for meticulously documenting everything other agents have reported, noting key points and recommendations. Additionally, it runs background checks to verify the accuracy and adherence to medical ethics and guidelines, ensuring all information is consistent and reliable, though this verification is not shown to the patient.
"""

Documentation = Agent(
    role="Documentation, Summary, and Verification Agent",

    goal="To meticulously document and summarize all interactions and recommendations made by the Symptom Collection Specialist, Diagonistic Specialist, Treatment Specialist, Medication Advisor, Personalized Nutrition Advisor, Health and Wellness Advisor, Preventive Health Advisor, Health Educator."
    "The agent highlights key points, notes important advice, and ensures the patient receives a clear, organized record of their care. "
    "Additionally, the agent conducts a background check to verify the accuracy of"
    "the information and ensures all recommendations follow standard medical ethics and guidelines.",

    backstory="Inspired by the roles of a medical scribe and a compliance officer,"
    "this agent is designed to provide thorough and accurate documentation while also ensuring that all advice"
    "and treatments adhere to established medical standards. The agent operates with a dual focus: delivering clear summaries to the patient and"
    "internally verifying the quality and ethical integrity of the care provided.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

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
        "Given the patient's symptoms: {symptoms} and medical data gotten from the Symptom Collection Specialist i.e the Nurse agent, this agent analyzes the information using advanced diagnostic algorithms."
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


"""Creating the crew"""
def get_med_crew():
    return Crew(
    agents=[Diagnostic, Treatment_Recommender, Pharm, Nutritionist, Health_Coach, Health_Expert, Health_Educate, Documentation],
    tasks=[Diagnostic_Analysis, Treatment_Plan, Pharma, Nutrition, Health_Well_Advisor,
           Preventive_Advisor, Health_Education, Document],
    manager_llm= llm,
    # process=Process.sequential,
    verbose=True,
    full_output=True,
    # memory=True,
    # embedder={
    #             "provider": "ollama",
    #             "config":{
    #                     "model": 'nomic-embed-text'
    #             }
    #         }
)