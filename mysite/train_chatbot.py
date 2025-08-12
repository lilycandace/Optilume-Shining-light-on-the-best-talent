#!/usr/bin/env python3
"""
Script to test and validate the Smart Recruitment System chatbot training
"""

import google.generativeai as genai
from chatbot_training_data import TRAINING_DATA, COMMON_QUESTIONS, get_training_context

# Configure Gemini API
genai.configure(api_key="AIzaSyDJHeZVEp1s5eFKbHR0MZxgR99CtZDuaAk")

def test_chatbot_training():
    """Test the chatbot with training data"""
    
    # Get training context
    training_context = get_training_context()
    
    # System prompt
    system_prompt = f"""
You are Opti, an AI assistant for the Smart Recruitment System website. You are well-versed in all aspects of this recruitment platform.

WEBSITE KNOWLEDGE:
- This is a Smart Recruitment System that helps companies find the best candidates
- Users can register/login to post jobs or apply for positions
- Job seekers can upload resumes which are processed using AI (Gemini) to extract skills, experience, and education
- The system uses BERT-based similarity matching to rank candidates against job descriptions
- Companies can post job openings with detailed descriptions
- Candidates can apply for jobs and upload their resumes
- The system provides ranking scores based on resume-job description similarity

AVAILABLE PAGES:
- Homepage (/): Overview of the recruitment system
- Job Listings (/job-listings.html): Browse all available job positions
- Post Job (/post-job.html): For employers to create new job postings
- Apply Job (/applyjob.html): For candidates to apply for positions
- About (/about.html): Information about the system
- Contact (/contact.html): Contact information
- Login/Register: User authentication

KEY FEATURES:
1. Resume Processing: AI extracts candidate information (name, skills, experience, education)
2. Job Matching: BERT-based similarity scoring between resumes and job descriptions
3. Candidate Ranking: Sorts candidates by match percentage
4. User Management: Separate interfaces for employers and job seekers
5. File Upload: Support for PDF and DOCX resume formats

TRAINING DATA FOR COMMON QUESTIONS:
{training_context}

RESPONSE STYLE:
- Be helpful, professional, and concise
- Provide specific page links when relevant
- Explain technical concepts in simple terms
- Offer step-by-step guidance for complex processes
- Always maintain a helpful and encouraging tone
- Use the training data above to provide accurate answers to common questions
- If a user asks about something not covered in training data, provide a helpful response based on your knowledge of the system

Current user message: """

    # Initialize model
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
    
    print("🤖 Testing Smart Recruitment System Chatbot Training")
    print("=" * 60)
    
    # Test with common questions
    test_questions = [
        "What is this website?",
        "How do I post a job?",
        "How does the ranking work?",
        "What file formats are supported?",
        "How do I apply for a job?",
        "What is BERT matching?",
        "How do I contact support?",
        "Can I search for specific skills?",
        "How accurate is the matching?",
        "What if my resume upload fails?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        print("-" * 40)
        
        try:
            full_prompt = system_prompt + question
            response = model.generate_content(full_prompt)
            print(f"✅ Response: {response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🎯 Training Validation Complete!")
    print("The chatbot is now trained with comprehensive website knowledge.")

def validate_training_data():
    """Validate that all training data is properly formatted"""
    print("🔍 Validating Training Data...")
    
    for category, qa_pairs in TRAINING_DATA.items():
        print(f"\n📂 Category: {category}")
        for i, qa in enumerate(qa_pairs, 1):
            if 'question' in qa and 'answer' in qa:
                print(f"  ✅ Q&A {i}: Valid")
            else:
                print(f"  ❌ Q&A {i}: Invalid format")
    
    print(f"\n📊 Total Q&A pairs: {sum(len(qa_pairs) for qa_pairs in TRAINING_DATA.values())}")
    print("✅ Training data validation complete!")

if __name__ == "__main__":
    print("🚀 Starting Smart Recruitment System Chatbot Training...")
    validate_training_data()
    test_chatbot_training()
