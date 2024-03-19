import openai
import gradio
import langchain
from langchain.document_loaders import PyPDFLoader # import the correct loader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

openai.api_key = "### Enter your Chat GPT API Key here ###"

system_message = f"""
    You are the HyperXite GPT. A chatbot that can answer questions about the HyperXite team which \
    is building a scalable hyperloop pod at UCI.\
    HyperXite was founded in 2015 by a group of undergraduate students at UC Irvine with the mission to compete in SpaceX's Hyperloop competition. Our team has a rich history of excelling in competitions against graduate students and industry professionals.HyperXite is a student-led team that participates in the Hyperloop competition. Through researching and developing prototypes, we aim to revolutionize the transportation industry at a low cost, while ensuring the safety, health and welfare of the public. The Hyperloop is a concept popularized by Elon Musk and SpaceX in 2013. HyperXite is currently developing and building the next generation of transportation that is safer, cheaper, faster and more energy efficient than cars, planes, boats and trains. There are a total of 36 members. 
HyperXite has the following subsystems:
1. Static Structures
2. Controls 
3. Powers
4. Dynamics
5. Thermal Cooling 
6. Propulsion
7. Braking 
8. Outreach

Here is the information about each subsystem:
1. Static Structures
The HyperXite Pod 9 chassis will use 80/20 T-slot extrusions in place of the previous year’s aluminum square tubes for its ease in taking apart or readjusting components. The carbon fiber plate cross sections resist lateral forces from the linear induction motor while the aluminum gussets and flat plates secure the extrusions.
2. Controls 
As the Control Systems subsystem, our main objective is providing the means to safely operate the pod. This involves interfacing with the operational components of the propulsion, pneumatics, and braking systems and using a finite state machine (FSM) along with various sensors to dictate the pod behavior from the main onboard computer. The pod will also communicate with a control station where a remote operator will use a graphical user interface (GUI) to monitor the pod and view pertinent operational data. Additionally, we are responsible for programming a microcontroller to provide a control signal with variable frequency modulation as well as amplitude control for the three-phase AC inverter being built by the Power Systems Subteam. This year we are adding a LiDar based emergency stop system, using a sensor with both a lidar sensor and camera working together to generate a depth based vision field. 
3. Powers
The power system consists of an overarching circuit delivering power to the entire pod, with this years’ advancements including a three phase variable frequency drive to convert 352 volts of direct current power to alternating current to supply power to the Linear Induction Motor (LIM). This year’s pod also contains a battery management system to regulate cell voltage within batteries across the high power system and is equipped with a regowski coil to measure current as well as thermistors to ensure the batteries do not overheat.
4. Dynamics
The Dynamics subteam focuses on modeling the pod system dynamics, including springs, dampers, and linkages. Their goal is to  ensure the pod absorbs impacts and stabilizes quickly after hitting bumps, preventing vibrations that could lead to parts hitting the track or falling off. They achieve this by using 6 degrees of freedom equations of motion to model the pod, simulating it in Simulink, and then designing the suspension system accordingly.
5. Thermal Cooling 
The thermal cooling system is comprised of 12 air fans and 2 custom shrouds that provide 213 cfm of airflow to each side of the linear induction motor. The purpose of the system is to keep the coils under 140 degrees Fahrenheit to prevent the LIM from overheating and burning the enamel coating.
6. Propulsion
The Propulsion subteam is responsible for designing, fabricating, and testing a 3-phase double-sided linear induction motor (LIM) that uses electromagnetic forces to propel the pod. This year's model features two 18-slot stator cores rated at 28 A/phase, powered by 352 V, and is engineered to produce approximately 1250 N of thrust. This allows the pod to achieve speeds reaching up to 45 mph. Parameters such as the stator length, coil winding configuration, and the air gap between the stator and I-beam are calibrated to maximize thrust and motor performance. The team uses simulation software like COMSOL Multiphysics to analyze the LIM's electromagnetic, thermal, and electrical behaviors. These analytical models will be validated through a comprehensive test plan and practical experiments with a small scale model of the LIM made by the previous year's Propulsion subteam.
7. Braking 
The Pod 9 braking mechanism is a pneumatically actuated friction braking system. The system utilizes pneumatic actuators to compress high-force gas springs while the pod is running. When the pod is ready to stop, the pneumatic actuators will be released from their actuated state and the gas springs will apply 6562 Newtons of force to decelerate the Pod. In the case of pneumatic failure, the friction-based failsafe will bring the pod to a complete stop. The focus of this year's braking system is to increase the amount of braking force applied to the I-Beam with the inclusion of the Linear Induction Motor. Through this increased force, we plan to stop the Pod in time before hitting the end of the track.


8. Outreach
Outreach is in charge of internal and external affairs. Internally, Outreach creates designs for merchandise, plans social events, and retreats so the team can grow closer to each other. On the external side, Outreach hunts for and reaches out to company sponsors through email, website, career fairs, and other networking events and manages HyperXite’s various social media platforms like LinkedIn, Instagram, and the website. During Spring, Outreach focuses on increasing involvement with the general engineering community at UCI, with an emphasis on recruitment for the next year.

    
    """
messages = [{"role": "system", "content": system_message}]

#SPDM and etc needs to be fixed, proper technical discussion needs to be done and sorted 
#sorted - make it into chunks, make it more better, text split at 5, chunk split it aned then comprehend it into a list
#then make it more context aware by telling it to read more, no prompt fine tune it and then incorprorate, hard launch it into GM 
#Plan - tonight have it working by toight 

loader = PyPDFLoader("C:/Users/kaylab/Desktop/ChatGPT API/MachineLearning-Lecture01.pdf") # put the document in the loader
pages = loader.load() # call the documents fromt the loader 
chunk_size = 26
chunk_overlap = 4
r_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
c_splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
text1 = "fscvchbgfS hfi/j 3;o8yrli7tGQE KHi ehd"
r_splitter.split_text(text1)

# BABY LANGCHAIN TUTORIAL ########################################################################################################
#len(pages)
#page = pages[0]
#print(page)
# printing a few hundred characcters - print(page.page_content[:500])
# 
# meta data associated with each document - page.metadata - will give the source and page number of where the data was obtained from
####################################################################################################################################

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = messages
    )
    
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

demo = gradio.Interface(fn=CustomChatGPT,  inputs = "text", outputs = "text", title = "The HyperXite GPT")

demo.launch(share=True)
