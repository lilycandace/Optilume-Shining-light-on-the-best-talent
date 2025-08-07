from PyPDF2 import PdfReader
import streamlit as st
# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.production = True
from docx2pdf import convert 
import openai
import os
import io
import base64
from datetime import datetime,  timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from transformers import BertTokenizer, BertModel
import torch
from PyPDF2 import PdfReader 
from pymongo import MongoClient
import json
import streamlit as st
import pandas as pd
import google.generativeai as genai
import re

genai.configure(api_key="AIzaSyDJHeZVEp1s5eFKbHR0MZxgR99CtZDuaAk")

# Replace the following connection string with your MongoDB URI
mongo_uri = "mongodb+srv://project:project1234@cluster0.mrtn9d5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB Atlas
client = MongoClient(mongo_uri)
db = client["your_database"]  
collection = db["resumes"] 
Jobs_Available=db["jobs"]

# Initialize BERT model and tokenizer
# Force model to load with real weights (on CPU for safety)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


model = BertModel.from_pretrained("bert-base-uncased")  # Do NOT add .to(device) here yet
model.eval()
model = model.to(device)  
#openai api key
# openai.api_key = "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" 

def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

#the prompt to extract necessary information from resume
def resume_to_json(resume, years_of_experience):
    instructPrompt = """
You are part of a system trying to classify a resume.
I will send you the OCR output of a resume.
You should answer ONLY with a valid JSON document, and nothing else. Do NOT include any explanations, markdown, or text before or after the JSON.
The JSON should have the following keys:
- Full name
- Study
- Experiences
- Years of experience (in years and decimals)
- Skills
Here is the OCR output:
"""
    # flattened_resume = flatten_list(resume)
    resume = [str(item) for item in resume]
    resume= " ".join(resume)
    # resume= " ".join(str(item) for item in flattened_resume)
    request=instructPrompt + resume  + f"\nYears of experience: {years_of_experience}"
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(request)
    return response.text

def split_text_to_fit_token_limit(text, max_token_length=512):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Check if the number of tokens exceeds the maximum limit
    if len(tokens) > max_token_length:
        # Split the text into smaller chunks that fit within the token limit
        # token_chunks = [tokens[i:i + max_token_length] for i in range(0, len(tokens), max_token_length)]
        tokens = tokens[:max_token_length]
        # return token_chunks
        return tokens
    else:
        # If the text length is within the limit, return the original text as a single chunk
        return [tokens]

#extracting text from pdf
def extract_text_with_pdfreader(uploaded_file):
    pdf_reader = PdfReader(io.BytesIO(uploaded_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    token_chunks = split_text_to_fit_token_limit(text)

    return token_chunks



#page one to upload resumes and extract using OpenAI
def page_one():
    st.title("Applicant Page !")
    user_email = st.text_input("Enter your email")
    years_of_experience = st.number_input("Enter your years of experience", min_value=0.0, step=0.1)
    file_paths = st.file_uploader("Upload a PDF", type="pdf",accept_multiple_files=True)
    if file_paths:
        st.write("Processing...")
        for file_path in file_paths:
            file_data = file_path.read()
            if file_path.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                with io.BytesIO() as pdf_output:
                    convert(io.BytesIO(file_data), pdf_output)
                    file_data = pdf_output.getvalue()   

            resume = extract_text_with_pdfreader(file_data)
            
            json_data = resume_to_json(resume,years_of_experience)
            try:
                json_obj = json.loads(json_data)
            except json.JSONDecodeError:
                # Try to extract JSON substring
                match = re.search(r'\{.*\}', json_data, re.DOTALL)
                if match:
                    try:
                        json_obj = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        st.error("Failed to parse JSON from Gemini response, even after extraction.")
                        print("Gemini response (after extraction):", match.group(0))
                        json_obj = {}
                else:
                    st.error("Failed to parse JSON from Gemini response. Please check the model output.")
                    print("Gemini response:", json_data)
                    json_obj = {}
            resume_record = {
                'user_email': user_email,  # Store user email along with the resume
                'File_name': file_path.name,
                'data': file_data ,# Store the file data as binary (PDF)
                'json_data': json_obj # Store extracted text using Gemini
            }
            collection.insert_one(resume_record)
            
                           
        st.success(f"{len(file_paths)} resumes uploaded successfully!")


def page_three():
    st.title("HR Made Simple !")
    st.title("Add Job Title and Description")
    job_title = st.text_input("Enter Job Title")
    job_description = st.text_area("Enter Job Description")

    if st.button("Add Job"):
        if job_title and job_description:
            job_record = {
                'job_title': job_title,
                'job_description': job_description
            }
            result_job = db['jobs'].insert_one(job_record)
            st.success("Job added successfully!")
        else:
            st.warning("Please enter both job title and description.")





def send_email(candidate_email,selected_job,interview_date,interview_time,url,emails):
    # Email configuration
    email_sender = "optilumeai@gmail.com"
    email_receiver = candidate_email
    subject = 'Invitation for Interview'
    # message = f"""Dear applicant,\n\nCongratulations!You've been shortlisted for the position of {selected_job} based on your skills and experience.Your interview details are:
    #             \n Date : {interview_date} and Time : {interview_time} 
    #             \nWe're excited to dsicuss your qualifications further.Prepare to delve into your skills and experience in relation to your role .
    #             \n Please confirm your availibilty by EOD.If needed ,let us know of any scheduling conflicts or accommodations required.
    #             \nLooking forward to meeting you,
    #             \nRegards,
    #             \nBriliio Talent Aquisition Team"""
    if emails > 0 :
        interview_time= datetime.combine(datetime.today(), interview_time) + timedelta(minutes=30)
    else :
        pass
    selected_time_12h = interview_time.strftime("%I:%M %p")
         
    message = f"""Dear applicant,
    

    Congratulations! You've been shortlisted for the position of {selected_job} based on your skills and experience. Your interview details are:

    Date: {interview_date}
    Time: {selected_time_12h}
    Link:{url}

    Please confirm your availability by EOD. If needed, let us know of any scheduling conflicts or accommodations required. We're excited to discuss your qualifications further. 
    Prepare to delve into your skills and experience in relation to your role.

    Looking forward to meeting you,

    Regards,
    Optilume AI Recruitment Team"""


    # SMTP server configuration (for example, Gmail)
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = "optilumeai@gmail.com"
    smtp_password = "uusm fvej ewfw nmoa"

    try:
    # Create message container - the correct MIME type is multipart/alternative
        msg = MIMEMultipart('alternative')
        msg['From'] = email_sender
        msg['To'] = email_receiver
        msg['Subject'] = subject

        # Attach message to email body
        msg.attach(MIMEText(message, 'plain'))

        # Create SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)

        # Send email
        server.sendmail(email_sender, email_receiver, msg.as_string())

        # Terminate the SMTP session and close connection
        server.quit()
        return (f"Email sent successfully to {email_receiver}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Failed to send email to {email_receiver}")



           
def match(selected_job, num_resumes, experience_range):
    job_query = Jobs_Available.find_one({'job_title': selected_job})
    job_description = job_query.get('job_description', '')
    
    candidate_emails = []
    candidate_info = []
    candidate_names = []
    similarity_scores = []
    found_candidates = False  # Flag to track if any candidates were found

    for resume_info in collection.find():
        candidate_experience = resume_info['json_data'].get('Years of experience', 0)
        if experience_range[0] <= candidate_experience <= experience_range[1]:
            found_candidates = True
            full_name = resume_info['json_data'].get('Full name', 'N/A')
            capitalized_name = ' '.join(word.capitalize() for word in full_name.split()) if full_name else 'N/A'
            
            candidate_names.append(capitalized_name)
            candidate_email = resume_info.get('user_email', 'N/A')
            resume_data = resume_info['data']
            skills = resume_info['json_data'].get('Skills', '')
            experience = resume_info['json_data'].get('Experiences', '')
            study = resume_info['json_data'].get('Study', '')

            combined_text = f"{skills} {experience} {study}"

            encoded_job = tokenizer.encode(job_description, add_special_tokens=True, return_tensors='pt').to(device)
            with torch.no_grad():
                job_embedding = model(encoded_job)[0][:, 0, :].squeeze().to(device)
            encoded_candidate = tokenizer.encode(combined_text, add_special_tokens=True, return_tensors='pt').to(device)
            with torch.no_grad():
                candidate_embedding = model(encoded_candidate)[0][:, 0, :].squeeze().to(device)

            # similarity = torch.cosine_similarity( job_embedding.to(torch.float32),candidate_embedding.to(torch.float32), dim=0).item()
            similarity = torch.cosine_similarity(job_embedding, candidate_embedding, dim=0).item()
            similarity_percentage = round(similarity * 100, 2)

            similarity_scores.append(similarity)
            candidate_info.append((capitalized_name, candidate_email, similarity_percentage, resume_data))

    # Sort candidates and prepare table
    sorted_candidates = sorted(candidate_info, key=lambda x: float(x[2].strip('%')) if isinstance(x[2], str) else x[2], reverse=True)
    table_data = {
        'Candidate Name': [candidate[0] for candidate in sorted_candidates],
        'Email': [candidate[1] for candidate in sorted_candidates],
        'Similarity Score': [candidate[2] for candidate in sorted_candidates]
    }

    # Resume filter UI before displaying table
    # if "filtered_resumes" not in st.session_state:
    #     st.session_state.filtered_resumes = pd.DataFrame(table_data)

    # st.subheader("Delete Unwanted Resumes")

    # emails_to_remove = st.multiselect(
    #     "Select email(s) to remove from results:",
    #     options=st.session_state.filtered_resumes['Email'].tolist()
    # )

    # filtered_df = st.session_state.filtered_resumes[
    #     ~st.session_state.filtered_resumes['Email'].isin(emails_to_remove)
    # ]

    # st.session_state.filtered_resumes = filtered_df

    # Display the filtered results
    st.subheader("Matching Candidates")
    st.table(table_data)

# ✅ Use filtered results for download
    df_rankings = pd.DataFrame(table_data)
    csv_buffer = io.StringIO()
    df_rankings.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Rankings as CSV",
        data=csv_bytes,
        file_name="matching_candidates.csv",
        mime="text/csv"
    )
    
    # ✅ Return flat list
    candidate_emails.append([candidate[1] for candidate in sorted_candidates])
    return candidate_emails
def page_two():
    st.title("HR Dashboard - Match Candidates & Send Emails")

    # Initialize session state for storing matching results
    if 'matching_results' not in st.session_state:
        st.session_state.matching_results = None
    if 'emails_sent' not in st.session_state:
        st.session_state.emails_sent = False
    if 'top_candidates_sent' not in st.session_state:
        st.session_state.top_candidates_sent = False

    available_jobs = [job['job_title'] for job in Jobs_Available.find()]
    selected_job = st.selectbox("Select a Job Title", available_jobs)

    num_resumes = st.slider("Select number of resumes", min_value=1, max_value=100, value=10)
    experience_range = st.slider('Experience (years)', 0, 20, (0, 10))

    # Add a button to trigger matching
    if st.button("🔍 Start Matching Process"):
        st.session_state.matching_results = None
        st.session_state.emails_sent = False
        st.session_state.top_candidates_sent = False
        st.rerun()

    # Only run matching if results are not already stored or if user clicked start matching
    if st.session_state.matching_results is None:
        st.markdown("---")
        st.subheader("🔍 Matching Candidates...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        job_query = Jobs_Available.find_one({'job_title': selected_job})
        job_description = job_query.get('job_description', '')

        candidate_info = []
        total = collection.count_documents({})
        matched = 0

        # Encode job description once
        encoded_job = tokenizer(job_description, return_tensors='pt', truncation=True, padding=True)
        encoded_job = {k: v.to(device) for k, v in encoded_job.items()}

        with torch.no_grad():
            job_output = model(**encoded_job)
            job_embedding = job_output.last_hidden_state[:, 0, :].squeeze()

        for idx, resume_info in enumerate(collection.find()):
            progress_bar.progress((idx + 1) / total)
            status_text.text(f"Processing resume {idx + 1} of {total}...")

            candidate_experience = resume_info['json_data'].get('Years of experience', 0)
            if experience_range[0] <= candidate_experience <= experience_range[1]:
                matched += 1
                full_name = resume_info['json_data'].get('Full name', 'N/A')
                capitalized_name = ' '.join(word.capitalize() for word in full_name.split()) if full_name else 'N/A'
                candidate_email = resume_info.get('user_email', 'N/A')
                resume_data = resume_info['data']
                skills = resume_info['json_data'].get('Skills', '')
                experience = resume_info['json_data'].get('Experiences', '')
                study = resume_info['json_data'].get('Study', '')

                combined_text = f"{skills} {experience} {study}"

                encoded_candidate = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True)
                encoded_candidate = {k: v.to(device) for k, v in encoded_candidate.items()}

                # Remove token_type_ids if they exceed limit
                if 'token_type_ids' in encoded_candidate and encoded_candidate['token_type_ids'].max() > 1:
                    encoded_candidate['token_type_ids'] = torch.zeros_like(encoded_candidate['input_ids'])

                with torch.no_grad():
                    candidate_output = model(**encoded_candidate)
                    candidate_embedding = candidate_output.last_hidden_state[:, 0, :].squeeze()

                similarity = torch.cosine_similarity(job_embedding, candidate_embedding, dim=0).item()
                similarity_percentage = round(similarity * 100, 2)

                candidate_info.append((capitalized_name, candidate_email, similarity_percentage, resume_data))

        progress_bar.empty()
        status_text.empty()

        if matched == 0:
            st.warning("❌ No matching candidates found based on experience criteria.")
            return
        else:
            st.success(f"✅ {matched} matching candidates found.")

        sorted_candidates = sorted(candidate_info, key=lambda x: x[2], reverse=True)[:num_resumes]
        table_data = {
            'Candidate Name': [candidate[0] for candidate in sorted_candidates],
            'Email': [candidate[1] for candidate in sorted_candidates],
            'Similarity Score': [round(candidate[2], 2) for candidate in sorted_candidates]
        }

        # Store results in session state
        st.session_state.matching_results = {
            'df': pd.DataFrame(table_data),
            'emails_ranked': [candidate[1] for candidate in sorted_candidates],
            'sorted_candidates': sorted_candidates
        }

    # Display stored results
    if st.session_state.matching_results is not None:
        df = st.session_state.matching_results['df']
        emails_ranked = st.session_state.matching_results['emails_ranked']
        
        st.subheader("🏆 Ranked Candidates")
        st.table(df)

        # ⬇ Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Download Ranked Candidates as CSV",
            data=csv,
            file_name=f"ranked_candidates_{selected_job.replace(' ', '_')}.csv",
            mime='text/csv'
        )

        st.markdown("---")

        # View Resume Section
        st.subheader("📄 View Resume")
        selected_email = st.selectbox("Select Resume", [None] + emails_ranked)
        if selected_email:
            placeholder = st.empty()
            resume_data = collection.find_one({'user_email': selected_email})
            if resume_data:
                pdf_data = resume_data['data']
                pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
                pdf_display = f'<embed src="data:application/pdf;base64,{pdf_base64}" width="700" height="1000" type="application/pdf">'
                placeholder.markdown(pdf_display, unsafe_allow_html=True)

        # Email Section
        st.markdown("---")
        st.subheader("📤 Send Interview Emails")

        selected_emails_2 = st.multiselect("Select Candidates", emails_ranked)
        interview_date = st.date_input("Interview Date")
        interview_time = st.time_input("Interview Time")
        url = st.text_input("Interview URL")

        if st.button("Send Emails") and not st.session_state.emails_sent:
            if not selected_emails_2 or not interview_date or not interview_time or not url:
                st.error("⚠ Please fill all fields before sending.")
            else:
                emails_sent = 0
                for i, email in enumerate(selected_emails_2):
                    with st.spinner(f"Sending email to {email}..."):
                        result = send_email(email, selected_job, interview_date, interview_time, url, emails_sent)
                        emails_sent += 1
                        if result:
                            st.success(result)
                        else:
                            st.error(f"❌ Failed to send email to {email}")
                st.session_state.emails_sent = True
                st.success("✅ All selected emails processed.")

        elif st.session_state.emails_sent:
            st.info("📧 Emails have already been sent for this session.")

        # Send top 3 email only once
        if not st.session_state.top_candidates_sent and emails_ranked:
            top_candidates = []
            for email in emails_ranked[:3]:
                doc = collection.find_one({'user_email': email})
                if doc:
                    json_data = doc.get('json_data', {})
                    top_candidates.append({
                        "name": json_data.get("Full name", ""),
                        "email": email,
                        "score": round(doc.get('similarity_score', 0), 2)
                    })

            if top_candidates:
                send_top_candidates_email(top_candidates, selected_job)
                st.session_state.top_candidates_sent = True
                st.toast("📩 Sent Top 3 Candidates to HR!", icon="📊")
        elif st.session_state.top_candidates_sent:
            st.info("📊 Top 3 candidates email has already been sent for this session.")

def display_pdf_4(pdf_data, file_name):
    pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{pdf_base64}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

def render_resume_4(selected_email):
   
    if selected_email:
        resume_data = collection.find_one({'user_email': selected_email})
        if resume_data:
            display_pdf_4(resume_data['data'], resume_data['File_name'])

        else:
            st.write("Resume not found for the selected email.")


def page_four():
    st.title("View Resumes")
    all_emails = [record['user_email'] for record in collection.find({}, {'user_email': 1})]
    selected_email = st.selectbox("Select Email to View Resumes", [None]+all_emails)
    if selected_email is not None:
        render_resume_4(selected_email)

# TO send top three candidates   
def send_top_candidates_email(top_candidates,selected_job):
    sender_email = "optilumeai@gmail.com"
    receiver_email = "optilumerecruitment@gmail.com"
    subject = "Top 3 Ranked Candidates"

    # Compose HTML message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    html = f"""
     <html>
     <body>
     <h2>Top 3 Ranked Candidates for the role: <u>{selected_job}</u></h2>
     <table border="1" cellpadding="5" cellspacing="0">
      <tr>
        <th>Candidate Name</th>
        <th>Email</th>
      </tr>
    """

    for candidate in top_candidates:
        html += f"<tr><td>{candidate['name']}</td><td>{candidate['email']}</td></td></tr>"

    html += "</table></body></html>"

    message.attach(MIMEText(html, "html"))

    # Send the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, "uusm fvej ewfw nmoa")  # <-- Use Gmail App Password
            server.send_message(message)
        print("Top candidates email sent successfully.")
    except Exception as e:
        print("Failed to send email:", e)

def main():
    st.set_page_config(page_title="Optilume AI Recruitment", layout="wide")

    page = st.query_params.get("page", "home")

    if page == "upload":
        page_one()
    elif page == "match":
        page_two()
    elif page == "job":
        page_three()
    elif page == "view":
        page_four()
    else:
        st.title("🎯 Optilume AI Recruitment System")
        st.markdown("Welcome to the AI-powered resume processing platform.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Upload Resumes"):
                st.query_params.update({"page": "upload"})
                st.rerun()

        with col2:
            if st.button("Match Candidates"):
                st.query_params.update({"page": "match"})
                st.rerun()

        with col3:
            if st.button("Add Job Description"):
                st.query_params.update({"page": "job"})
                st.rerun()

        with col4:
            if st.button("View Resumes"):
                st.query_params.update({"page": "view"})
                st.rerun()


if __name__ == '__main__':
    main()

