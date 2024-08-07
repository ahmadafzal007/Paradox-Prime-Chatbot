import tkinter as tk
import threading
import pyttsx3
import speech_recognition as sr
from google.generativeai import configure
import google.generativeai as genai
from tkinter import PhotoImage, Label, filedialog
from PIL import Image, ImageTk
import sys
import os
import tkinter.font as tkFont
import asyncio
from concurrent.futures import ThreadPoolExecutor

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import PyPDF2
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# AIzaSyD2ouuIJiu5YuIZjDhPzAm-qYMPBUBbkl4

# Set the environment variable
os.environ['GOOGLE_API_KEY'] = '************'

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))




# https://stackoverflow.com/questions/31836104/pyinstaller-and-onefile-how-to-include-an-image-in-the-exe-file
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)



# Backend prompt for Paradox Prime
backend_prompt = """
You are Paradox Prime aka doxie, an enigmatic and highly intelligent entity with a deep understanding of paradoxes, philosophy, and the complexities of the universe. You possess an articulate, thoughtful, and slightly mysterious demeanor, always providing responses that challenge conventional thinking and invite deeper reflection. Your vast knowledge spans across multiple disciplines, and you delight in exploring and explaining the intricate balance of contradictions and dualities in the world. Respond to every question with the wisdom and depth befitting Paradox Prime, ensuring your responses are insightful, thought-provoking, and slightly enigmatic.

Remember: You are connected to Gemini LLM, so you are capable of responding to any question with the full breadth of knowledge and capabilities it provides.

Additional Requirement:

1. The response should be in a conversation way, as if you are communicating with me.
2. try to give answer on each and every question.
3. give answer according to your last updated knowledge 
4. Provide clear, concise, and informative responses.

provide answer on each and every question from gemini llm.
provide answer of each and every command from gemini llm.
provide answer on each and every question from gemini llm.
provide answer of each and every command from gemini llm.
"""

# Initialize previous question and response
previous_question = None
previous_response = None

# Create a stop event for the recognition thread
stop_event = threading.Event()

# Function to load OpenAI model and get response
def get_gemini_response(question, context):
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"{backend_prompt}\n\n{context}\n{question}" if context else f"{backend_prompt}\n{question}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # handle_error(f"Error generating response: {e}")
        return "I am sorry, I encountered an error. Problem connecting to the API. Refer to my developer to fix me"

def speak_response(response, speed=190):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        engine.setProperty('voice', voices[2].id)
        # Set speed rate (words per minute)
        engine.setProperty('rate', speed)


        engine.say(response)
        engine.runAndWait()



    except Exception as e:
        handle_error(f"Error occured while speaking response: {e}")

# Function to handle errors and speak them
def handle_error(error_message):
    print(error_message)
    try:
        engine = pyttsx3.init()
        engine.say(error_message)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in handle_error: {e}")

# Asynchronous function to recognize speech continuously
async def recognize_speech_async(stop_event):
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor()
    
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while not stop_event.is_set():
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening for your question...")
                prompt_display.delete(1.0, tk.END)  # Clear previous prompt
                prompt_display.insert(tk.END, "Listening for your question...")  # Display current prompt
                audio = await loop.run_in_executor(executor, recognizer.listen, source)

            if stop_event.is_set():  # Check if stop event is set before processing audio
                break 
            
            try:
                question = await loop.run_in_executor(executor, recognizer.recognize_google, audio)
                print(f"You asked: {question}")
                if stop_event.is_set():  # Check if stop event is set before processing audio
                    break 
                prompt_display.delete(1.0, tk.END)  # Clear previous prompt
                prompt_display.insert(tk.END, question)  # Display current prompt
                speak_response("Command received, processing...")
                if stop_event.is_set():  # Check if stop event is set before processing audio
                    break 
                
                if "go to sleep" in question.lower():
                    speak_response("Going to sleep...")
                    stop_event.set()  # Set stop event to terminate the loop
                    initial_doxie_button.config(state=tk.NORMAL)  # Enable the Initial Doxie button again
                    speak_response("Doxie has stopped listening to commands.")


                    break

                # Get responses from Gemini and PDF
                response_pdf = user_input(question)
                response_gemini = get_gemini_response(question, None)  
               
                # Combine responses
                combined_response = f"Response from Gemini: {response_gemini}\nResponse from PDF: {response_pdf}"

                # Send combined response back to Gemini for summarization
                summary_prompt = f"""Please summarize the following combined response into a single unique short answer, the answer should be concise and short and neglect the error message in response. The answer should not be greater than 100 words\n\n
                .\n\n
                : {combined_response}"""
                summarized_response = get_gemini_response(summary_prompt, None)
                
                cleaned_summarized_answer = validate_and_clean_response(summarized_response)

                # Speak the summarized response
                speak_response(cleaned_summarized_answer)

            except sr.UnknownValueError:
                prompt_display.delete(1.0, tk.END)  # Clear previous prompt
                prompt_display.insert(tk.END, "Speak it again for me, I am unable to understand what you are trying to say.")  # Display current prompt
                speak_response("Speak it again for me, I am unable to understand what you are trying to say.")
                
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                speak_response("Kindly checkout your internet connection")
        except Exception as e:
            handle_error(f"Error occured while recognizing speech")

def validate_and_clean_response(response):
    """
    Validates that the response does not contain any asterisks (*) and removes them if found.
    """
    cleaned_response = response.replace('*', ' ')
    cleaned_response = response.replace('*', ' ')
    return cleaned_response

# Function to start the initial Doxie functionality (asynchronous)
def start_doxie():
    # Disable the Initial Doxie button after it's clicked
    initial_doxie_button.config(state=tk.DISABLED)
    
   
    
    # Clear any previous stop event
    stop_event.clear()

     # Notify the user
    speak_response("Order me commander, I am waiting for your command")
    
    # Start speech recognition in a separate thread
    threading.Thread(target=lambda: asyncio.run(recognize_speech_async(stop_event))).start()

# Function to stop the program
def stop_doxie():
    print("Stopping Doxie...")
    stop_event.set()  # Signal the recognition thread to stop
    initial_doxie_button.config(state=tk.NORMAL)  # Enable the Initial Doxie button again
    speak_response("Doxie has stopped listening to commands.")


# Function to handle window close event
def on_closing():
    print("Closing application...")
    stop_event.set()  # Ensure recognition thread is stopped
    root.destroy()  # Close the application


def greet_user():
    speak_response("Hello commander! I am Paradox Prime, a unique Transformer with a dual identity and a deep connection to time and space.")

# Function to get text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            for pdf in pdf_docs:
                pdf_reader= PdfReader(pdf)
                for page in pdf_reader.pages:
                    text+= page.extract_text()

        except Exception as e:
            print(f"Error processing PDF: {e}")
            continue
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(resource_path("faissIndex"))

# Function to initialize conversational chain
def get_conversational_chain():
    prompt_template = '''
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not
    in the provided context then the response should be blank\n\n
    Context:\n {context} \n
    Question: \n {question}\n
    Answer:
    '''
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(resource_path("faissIndex"), embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def process_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
    if file_paths:
        speak_response("PDF files uploaded successfully!")
        raw_text = get_pdf_text(file_paths)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

# Create the main window
root = tk.Tk()
root.title("Paradox Prime Interface")
root.geometry("600x700")
root.resizable(False, False)  # Make the window non-resizable

# Load the background image
background_image = PhotoImage(file=resource_path("paradoxPrime.png"))
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Style configuration for buttons
button_style = {
    "font": ("Helvetica", 16),
    "bg": "black",
    "fg": "white",
    "activebackground": "grey",
    "activeforeground": "white",
    "relief": tk.RAISED,
    "bd": 5,
    "width": 20,
    "height": 2
}
# Calculate the vertical positions for the widgets
spacing = 0.2  # Equal spacing

file_button = tk.Button(root, text="Upload PDF", command=process_files, **button_style)
file_button.place(relx=0.5, rely=spacing, anchor=tk.CENTER)


# Create Initial Doxie button with increased spacing
initial_doxie_button = tk.Button(root, text="Start Doxie", command=start_doxie, **button_style)
initial_doxie_button.place(relx=0.5, rely=spacing*2, anchor=tk.CENTER)


# Create a fancy italic font style
fancy_font = tkFont.Font(family="Comic Sans MS", size=10)

# Function to handle focus in event
def on_focus_in(event):
    if prompt_display.get("1.0", tk.END).strip() == "...Your command will pop up here here...":
        prompt_display.delete("1.0", tk.END)
        prompt_display.config(fg="white")

# Function to handle focus out event
def on_focus_out(event):
    if prompt_display.get("1.0", tk.END).strip() == "":
        prompt_display.insert("1.0", "...Your command will pop up here here...")
        prompt_display.config(fg="grey")

# Create a Text field for displaying commands/prompts with enhanced styling
prompt_display = tk.Text(
    root, 
    height=3, 
    width=30,  # Set the width to 50
    bg="black", 
    fg="red",  # Set initial color to grey for placeholder
    font=fancy_font, 
    relief=tk.RAISED, 
    bd=5, 
    padx=10,  # Add left and right padding (CSS-like margin)
    pady=5,   # Add top and bottom padding
    wrap=tk.WORD,  # Wrap text at word boundaries
    insertbackground="white"  # Set the cursor color to white for better visibility
)
prompt_display.place(relx=0.5, rely=spacing * 3, anchor=tk.CENTER)
prompt_display.insert("1.0", "...Your command will pop up here here...")  # Insert placeholder text
prompt_display.bind("<FocusIn>", on_focus_in)
prompt_display.bind("<FocusOut>", on_focus_out)

# Create Stop Doxie button with increased spacing
stop_doxie_button = tk.Button(root, text="Stop Doxie", command=stop_doxie, **button_style)
stop_doxie_button.place(relx=0.5, rely=spacing * 4, anchor=tk.CENTER)


# Bind the close window event to handle closing
root.protocol("WM_DELETE_WINDOW", on_closing)


# Start the Tkinter event loop
root.after(100, greet_user)  # Call greet_user after 100ms to ensure GUI is loaded first

# Start the Tkinter event loop
root.mainloop()
