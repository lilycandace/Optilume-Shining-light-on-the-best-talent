"""
Training data for the Smart Recruitment System chatbot
This file contains Q&A pairs to train the Gemini AI chatbot
"""

TRAINING_DATA = {
    "website_overview": [
        {
            "question": "What is this website?",
            "answer": "This is a Smart Recruitment System that helps companies find the best candidates efficiently. It uses AI-powered resume processing and BERT-based matching to rank candidates against job descriptions."
        },
        {
            "question": "How does the system work?",
            "answer": "The system works in 4 main steps: 1) Employers post job openings, 2) Candidates upload resumes, 3) AI (Gemini) extracts candidate information, 4) BERT algorithm matches resumes to job descriptions and ranks candidates by similarity score."
        }
    ],
    
    "user_roles": [
        {
            "question": "What can employers do?",
            "answer": "Employers can: register/login, post job openings with detailed descriptions, view candidate applications, and get AI-ranked candidate lists based on resume-job matching scores."
        },
        {
            "question": "What can job seekers do?",
            "answer": "Job seekers can: register/login, browse job listings, apply for positions, upload resumes (PDF/DOCX), and track their applications."
        }
    ],
    
    "resume_processing": [
        {
            "question": "How does resume processing work?",
            "answer": "When you upload a resume, our AI (Gemini) automatically extracts: full name, education background, work experience, years of experience, and skills. This information is then used for job matching."
        },
        {
            "question": "What file formats are supported for resumes?",
            "answer": "The system supports PDF and DOCX (Word) file formats for resume uploads. Make sure your resume is in one of these formats before uploading."
        },
        {
            "question": "How accurate is the resume parsing?",
            "answer": "Our AI uses advanced natural language processing to extract information with high accuracy. However, it works best with well-formatted, clear resumes. The extracted data is used for matching but you can always verify the information."
        }
    ],
    
    "job_matching": [
        {
            "question": "How does the ranking system work?",
            "answer": "The ranking uses BERT (Bidirectional Encoder Representations from Transformers) to compare resume content with job descriptions. It analyzes skills, experience, and education to calculate similarity scores, then ranks candidates by percentage match."
        },
        {
            "question": "What is BERT matching?",
            "answer": "BERT is an AI model that understands context and meaning in text. It converts both job descriptions and resumes into numerical representations, then calculates how similar they are. Higher similarity scores mean better matches."
        },
        {
            "question": "How accurate is the matching?",
            "answer": "The BERT-based matching is quite accurate as it understands context and meaning, not just keywords. However, it's designed to assist human recruiters, not replace them entirely. Always review top candidates personally."
        }
    ],
    
    "navigation_help": [
        {
            "question": "How do I post a job?",
            "answer": "To post a job: 1) Register/login as an employer, 2) Go to the 'Post Job' page, 3) Fill in job details (title, description, requirements), 4) Submit. Your job will appear in the job listings for candidates to find."
        },
        {
            "question": "How do I apply for a job?",
            "answer": "To apply for a job: 1) Register/login as a candidate, 2) Browse job listings, 3) Click on a job you're interested in, 4) Click 'Apply' and upload your resume, 5) Fill in any additional information required."
        },
        {
            "question": "Where can I see all available jobs?",
            "answer": "Visit the 'Job Listings' page to see all available positions. You can browse through different job categories and search for specific roles or companies."
        },
        {
            "question": "How do I register?",
            "answer": "Click on the 'Register' link in the navigation menu. You'll need to provide basic information like name, email, and password. Choose whether you're registering as an employer or job seeker."
        }
    ],
    
    "technical_support": [
        {
            "question": "What if my resume upload fails?",
            "answer": "Make sure your file is in PDF or DOCX format and under 10MB. If it still fails, try converting your resume to PDF format, as this is the most reliable format for our system."
        },
        {
            "question": "How do I contact support?",
            "answer": "Visit the 'Contact' page for support information. You can find contact details and a contact form there to reach our support team."
        },
        {
            "question": "Is my data secure?",
            "answer": "Yes, we take data security seriously. Your personal information and resumes are stored securely, and we follow industry best practices for data protection. We only use your data for job matching purposes."
        }
    ],
    
    "advanced_features": [
        {
            "question": "Can I search for specific skills?",
            "answer": "The system automatically analyzes skills from resumes and job descriptions. When you post a job, include the required skills in the description. The AI will match candidates based on skill similarity."
        },
        {
            "question": "How do I know if I'm a good match for a job?",
            "answer": "The system provides similarity scores for each candidate. Higher percentages indicate better matches. However, these scores are guidelines - always apply if you're interested, as the AI may not capture all relevant experience."
        },
        {
            "question": "Can I edit my profile?",
            "answer": "Yes, after logging in, you can update your profile information, including contact details and preferences. This helps ensure accurate matching and communication."
        }
    ]
}

def get_training_context():
    """Returns formatted training data for the chatbot"""
    context = "TRAINING DATA FOR SMART RECRUITMENT SYSTEM CHATBOT:\n\n"
    
    for category, qa_pairs in TRAINING_DATA.items():
        context += f"CATEGORY: {category.replace('_', ' ').title()}\n"
        for qa in qa_pairs:
            context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
    
    return context

# Common user questions and their expected responses
COMMON_QUESTIONS = [
    "What is this website?",
    "How do I post a job?",
    "How do I apply for a job?",
    "How does the ranking work?",
    "What file formats are supported?",
    "How do I register?",
    "Where can I see all jobs?",
    "What is BERT matching?",
    "How accurate is the matching?",
    "How do I contact support?"
]
